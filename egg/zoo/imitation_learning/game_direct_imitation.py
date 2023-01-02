import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import argparse
from collections import defaultdict
from typing import Callable, Iterable
import torch.multiprocessing as mp

import egg.core as core
from egg.core import Callback, Interaction, InteractionSaver
from egg.core.baselines import Baseline, MeanBaseline
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths

from egg.zoo.imitation_learning.bc_archs import bc_agents_setup
from egg.zoo.imitation_learning.archs import define_agents


class DirectImitationGame(nn.Module):
    """
        Training class for the multitask: (1) original task + (2) imitation task.
        """

    def __init__(self,
                 senders: Iterable[nn.Module],
                 receivers: Iterable[nn.Module],
                 loss: Callable,
                 opts: argparse.Namespace,
                 baseline_type: Baseline = MeanBaseline,
                 train_logging_strategy: LoggingStrategy = None,
                 test_logging_strategy: LoggingStrategy = None,
                 ablation: str = 'all',
                 sender_aware_weight: float = 0.0):
        super(DirectImitationGame, self).__init__()

        self.opts = opts
        self.device = opts.device
        self.sample_pairs = opts.sample_pairs
        self.train_expert_sender = ablation in ['all', 'sender_only']
        self.train_expert_receiver = ablation in ['all', 'receiver_only']

        self.receiver_reinforce = opts.receiver_reinforce
        self.imitation_reinforce = opts.imitation_reinforce

        self.imitation_weight = self.opts.imitation_weight
        self.sender_aware_weight = sender_aware_weight
        self.sender_entropy_coeff = opts.sender_entropy_coeff
        self.receiver_entropy_coeff = 0.0
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

        # nn modules
        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        assert len(self.senders) == len(self.receivers)
        self.number_of_pairs = len(self.senders)
        self.comm_pairings = None
        self.imitation_pairings = None

        self.baselines = [defaultdict(baseline_type) for _ in range(self.number_of_pairs)]
        self.comm_loss = loss

    def sample_pairings(self, derangement=False):
        # https://stackoverflow.com/questions/26554211/numpy-shuffle-with-constraint
        def random_derangement(N):
            original = np.arange(N)
            new = np.random.permutation(N)
            same = np.where(original == new)[0]
            while len(same) != 0:
                swap = same[np.random.permutation(len(same))]
                new[same] = new[swap]
                same = np.where(original == new)[0]
                if len(same) == 1:
                    swap = np.random.randint(0, N)
                    new[[same[0], swap]] = new[[swap, same[0]]]
            return new
        if derangement: return random_derangement(self.number_of_pairs)
        pairs = list(range(self.number_of_pairs))
        if self.sample_pairs: random.shuffle(pairs)
        return pairs

    def init_pairings(self):
        self.comm_pairings = self.sample_pairings()  # sender i is paired with receiver comm_pairings[j]
        self.imitation_pairings = self.sample_pairings(derangement=True)

    def evaluate_on_game(self, sender, receiver, sender_input, labels, receiver_input=None, aux_input=None):
        message, log_prob_s, entropy_s, probas_s = sender(sender_input, aux_input)
        message_length = find_lengths(message)
        receiver_output, log_prob_r, entropy_r, rcvr_aux = receiver(
            message, receiver_input, aux_input, message_length
        )
        # Get communication loss
        comm_loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )
        # print('ACC: ', aux_info['acc'].shape)
        # print('ACC OR: ', aux_info['acc_or'].shape)
        effective_entropy_s = entropy_s[:, :self.opts.max_len].mean(dim=1)
        effective_log_prob_s = log_prob_s.sum(dim=1)
        effective_log_prob_r = log_prob_r.sum(dim=1)

        aux_info['loss'] = comm_loss.detach()
        aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()
        aux_info["length"] = message_length.float()  # will be averaged
        aux_info["probas_s"] = probas_s.detach()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            receiver_sample=rcvr_aux['sample'],
            aux_input=aux_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )
        return comm_loss, interaction, effective_entropy_s, effective_log_prob_s, effective_log_prob_r, probas_s, receiver_output

    def compute_imitation_loss(self, pairings, batch_size, interactions, log_probs, logits, agent_type: str):
        assert agent_type in ['sender', 'receiver']

        imi_losses = [torch.zeros(batch_size) for _ in range(self.number_of_pairs)]

        for i in range(0, len(pairings), 2):
            a, b = pairings[i], pairings[i + 1]
            if agent_type == 'sender':
                imi_losses[a] = F.cross_entropy(
                    logits[a], # remove the end token
                    interactions[b].message.to(self.device)[:,:self.opts.max_len],
                    reduction='none'
                ).view(batch_size, self.opts.max_len).mean(dim=-1)
                imi_losses[b] = F.cross_entropy(
                    logits[b],
                    interactions[a].message.to(self.device)[:,:self.opts.max_len],
                    reduction='none'
                ).view(batch_size, self.opts.max_len).mean(dim=-1)

            elif agent_type == 'receiver':
                receiver_output_a = logits[a].view(
                        batch_size, self.opts.n_attributes, self.opts.n_values
                    ).transpose(1, 2).to(self.device)
                receiver_output_b = logits[b].view(
                        batch_size, self.opts.n_attributes, self.opts.n_values
                    ).transpose(1, 2).to(self.device)

                imi_losses[a] = F.cross_entropy(
                    receiver_output_a,
                    interactions[b].receiver_sample.to(self.device),
                    reduction='none'
                ).view(batch_size, self.opts.n_attributes).mean(dim=-1)

                imi_losses[b] = F.cross_entropy(
                    receiver_output_b,
                    interactions[a].receiver_sample.to(self.device),
                    reduction='none'
                ).view(batch_size, self.opts.n_attributes).mean(dim=-1)

        if self.imitation_reinforce:
            imitation_policy_loss = [
                (imi_losses[i].detach() - self.baselines[i]['{}_imit_loss'.format(agent_type)].predict(imi_losses[i].detach())) *
                log_probs[i]
                for i in range(self.number_of_pairs)
            ]
            loss = torch.sum(torch.stack(imitation_policy_loss, dim=0))
        else:
            # direct backprop
            loss = torch.sum(torch.stack(imi_losses, dim=0))

        return loss, imi_losses

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        batch_size = sender_input.size(0)
        # print(batch_size)
        r_imitation_policy_loss = torch.zeros(batch_size).to(self.device)
        s_imitation_policy_loss = torch.zeros(batch_size).to(self.device)
        imi_r_losses = [torch.zeros(batch_size).to(self.device) for _ in range(self.number_of_pairs)]
        imi_s_losses = [torch.zeros(batch_size).to(self.device) for _ in range(self.number_of_pairs)]

        # Evaluate the pairs, produce the comm and imi losses
        comm_losses, interactions, speaker_entropies, log_prob_ss, log_prob_rs, logits_ss, logits_rs = [], [], [], [], [], [], []
        # print(163)

        # print(165)
        # global eval_
        # def eval_(i):
        #     j = self.comm_pairings[i]
        #     return self.evaluate_on_game(self.senders[i], self.receivers[j], sender_input, labels, receiver_input, aux_input)
        #
        # pool = mp.Pool(self.number_of_pairs)
        # comm_losses, interactions, speaker_entropies, log_prob_ss, log_prob_rs, logits_ss, logits_rs = \
        #     zip(*pool.map(eval_, range(self.number_of_pairs)))

        for i in range(self.number_of_pairs):
            j = self.comm_pairings[i]
            comm_loss, interaction, ent_s, log_prob_s, log_prob_r, logits_s, logits_r = self.evaluate_on_game(
                self.senders[i], self.receivers[j], sender_input, labels, receiver_input, aux_input
            )
            # print(171)
            comm_losses.append(comm_loss.to(self.device))
            interactions.append(interaction.to(self.device))
            speaker_entropies.append(ent_s.to(self.device))
            log_prob_ss.append(log_prob_s.to(self.device))
            log_prob_rs.append(log_prob_r.to(self.device))
            logits_ss.append(logits_s)
            logits_rs.append(logits_r)

        # Concatenates results from individual agents.
        interaction = Interaction.from_iterable(interactions)
        interaction.aux['pairings'] = torch.Tensor(self.comm_pairings)

        # Compute game policy losses for sender
        game_policy_losses = [(comm_losses[i].detach() - self.baselines[i]['comm_loss'].predict(comm_losses[i].detach()))\
                              * log_prob_ss[i] for i in range(self.number_of_pairs)]
        # print(185)
        # Allow the two pairs' senders to imitate each other using REINFORCE, produce this loss.
        if self.train_expert_sender or self.train_expert_receiver:
            if self.train_expert_receiver:
                r_imitation_policy_loss, imi_r_losses = self.compute_imitation_loss(
                    self.imitation_pairings, batch_size, interactions, log_prob_rs, logits_rs, 'receiver'
                )
                # print(194)
            if self.train_expert_sender:
                s_imitation_policy_loss, imi_s_losses = self.compute_imitation_loss(
                    self.imitation_pairings, batch_size, interactions, log_prob_ss, logits_ss, 'sender'
                )
            interaction.aux['imitation/pairings'] = torch.Tensor(self.imitation_pairings)
            interaction.aux['imitation/r_losses'] = torch.cat(imi_r_losses)
            interaction.aux['imitation/s_losses'] = torch.cat(imi_s_losses)

        # Compute policy loss with entropy term.
        entropy_term = sum([ent_s.mean() for ent_s in speaker_entropies]) * self.sender_entropy_coeff
        imitation_policy_loss = (r_imitation_policy_loss + s_imitation_policy_loss).mean()
        game_policy_loss = torch.sum(torch.stack(game_policy_losses, dim=0))
        r_game_loss = sum([comm_loss.mean() for comm_loss in comm_losses])

        optimized_loss = self.imitation_weight * imitation_policy_loss + \
                         game_policy_loss + r_game_loss - entropy_term

        if self.training:
            for i in range(self.number_of_pairs):
                self.baselines[i]["receiver_imit_loss"].update(imi_r_losses[i])
                self.baselines[i]["sender_imit_loss"].update(imi_s_losses[i])
                self.baselines[i]["comm_loss"].update(comm_losses[i])

        # print(interaction)
        # print('end of forward pass')
        return optimized_loss, interaction
