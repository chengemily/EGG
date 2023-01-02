import torch
import torch.nn as nn
import argparse
import os
import copy
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import egg.core as core
from egg.core.early_stopping import *
from egg.core.language_analysis import editdistance
from egg.core.util import find_lengths

from egg.zoo.imitation_learning.archs import Sender
from egg.zoo.imitation_learning.imitator_agent_with_forcing import RnnSenderReinforceImitator
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.train import *

from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.behavioural_cloning import *
from egg.zoo.imitation_learning.util import *
from egg.zoo.imitation_learning.loss import DiffLoss
from egg.zoo.imitation_learning.stock_senders import *
from egg.zoo.imitation_learning.analysis.direct_imitation_analysis import get_levenstein_distance


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, choices=['kl', 'cross_entropy'], default='cross_entropy')
    parser.add_argument("--n_attributes",
                        type=int,
                        default=2, help="")
    parser.add_argument('--reinforce',
                        type=int,
                        default=1, choices=[0, 1])
    parser.add_argument("--n_values",
                        type=int,
                        default=2, help="")
    parser.add_argument("--data_scaler", type=int, default=10)
    parser.add_argument('--hidden',
                        type=int,
                        default=50,
                        help='Size of hidden layer of both agents')
    parser.add_argument("--sender_cell", type=str, default="rnn")
    parser.add_argument("--receiver_cell", type=str, default="rnn")
    parser.add_argument("--variable_message_length", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--sender_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--convergence_epsilon",
        type=float,
        default=0.03,
        help="Stop training when gradient norm is less than epsilon."
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.1,
        help="Entropy weight"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Set True if you want model to be saved",
    )
    parser.add_argument(
        "--experts",
        nargs="+",
        type=list,
        default=['Compositional', 'Noncompositional']
    )
    parser.add_argument(
        "--expert_lengths",
        nargs="*",
        type=list,
        default=[]
    )

    args = core.init(arg_parser=parser, params=params)

    return args


class Imitator(nn.Module):
    def __init__(
            self, agent, opts, device, num_senders
    ):
        super(Imitator, self).__init__()
        self.agent = agent
        self.opts = opts
        self.use_reinforce = opts.reinforce
        self.variable_message_length = opts.variable_message_length
        self.device = device
        self.to(device)
        self.mean_baseline = [torch.Tensor([0.0]).to(self.device)] * num_senders
        self.counter = 0
        self.entropy_weight = opts.entropy_weight

    def update_baseline(self, loss, i):
        self.counter += 1
        self.mean_baseline[i] += (loss.detach().mean(dim=-1).mean() - self.mean_baseline[i]) / self.counter

    def get_reinforce_loss(self, loss, log_probs, expert_id, entropy):
        rf_loss = ((loss.detach() - self.mean_baseline[expert_id]) * log_probs)
        if self.training: self.update_baseline(loss, expert_id)

        return rf_loss - self.entropy_weight * entropy

    def forward(self, sender_input, ground_truth):
        sender_input = sender_input.to(self.device)
        sender_output, log_prob_s, entropy_s, class_proba_s = self.agent(sender_input, ground_truth)
        batch_size = sender_output.shape[0]
        class_proba_s = class_proba_s.reshape(
            batch_size * self.opts.max_len, self.opts.vocab_size + 1
        )

        return sender_output, log_prob_s, entropy_s, class_proba_s, batch_size

    def get_edit_distance(self, m1, m2):
        return get_levenstein_distance(m1, m2)

    def mask_sender_output(self, message):
        """
        Returns a sender output mask (1's until the message length, then 0's filled in for the rest).
        """
        if self.variable_message_length:
            sender_lengths = find_lengths(message)
            batch_size = message.shape[0]
            sender_output_mask = (
                    torch.arange(0, self.opts.max_len, dtype=torch.float32).repeat(batch_size, 1).to(self.device)
                    < sender_lengths.unsqueeze(1)).long()
            return sender_output_mask

        return torch.ones(message.shape)

    def score(self, sender_input, experts, print_outputs=False):
        """
        Returns cross-entropy loss between the output of the BC sender and the output
        of the expert sender given the same input.
        :param interaction: (Interaction)
        :param val: (bool) whether in evaluation mode
        :return: cross entropy loss
        """
        losses, accs, entropies = [], [], []
        aux = {}

        if not self.training:
            sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(sender_input, None)
            sender_output = sender_output[:, :self.opts.max_len]  # cut off end token

        for i, expert in enumerate(experts):
            message = expert.forward(sender_input).to(self.device)
            message = message[:, :self.opts.max_len]
            aux[expert.name + '_expert_message'] = message

            if self.training:
                sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(sender_input, message)
                sender_output = sender_output[:, :self.opts.max_len]  # cut off end token

            # Convert end of sender output to 0s
            aux[expert.name + '_imitator_message'] = sender_output

            if print_outputs:
                print(torch.exp(log_prob_s[:10, :-1]))
                print('sender output: ', sender_output[:10])
                print('expert out: ', message[:10])

            # These are log probabilities
            class_proba_s = class_proba_s.reshape((
                batch_size, self.opts.max_len, self.opts.vocab_size + int(self.opts.variable_message_length)
            )).transpose(1, 2)

            sender_output_mask = self.mask_sender_output(sender_output)

            if self.use_reinforce:
                # Negative accuracy
                acc_or = (
                        sender_output == message
                ).float()

                # only give the loss to sender message up to (incl) EOS
                s_loss = self.get_reinforce_loss(-acc_or, log_prob_s[:,:-1], i, entropy_s[:,:-1]) * sender_output_mask
            else:
                s_loss = F.nll_loss(class_proba_s, message, reduction='none') * sender_output_mask

            # s_loss = s_loss.sum(dim=-1).mean()
            s_loss = s_loss.mean(dim=-1).mean()
            accuracy = (torch.eq(sender_output, message) * sender_output_mask).sum() / sender_output_mask.sum()
            accuracy_message = ((torch.eq(sender_output, message) * sender_output_mask).sum(dim=-1) \
                                == int(self.opts.expert_lengths[i])).sum() / batch_size
            losses.append(s_loss)
            accs.append(accuracy)
            entropies.append(entropy_s.detach().mean().item())

            aux[expert.name + '_acc'] = torch.Tensor([accuracy])
            aux[expert.name + '_acc_message'] = torch.Tensor([accuracy_message])
            aux[expert.name + '_loss'] = torch.Tensor([s_loss.detach()])
            aux[expert.name + '_entropy'] = torch.Tensor([entropy_s.detach().mean()])
            aux[expert.name + '_class_probs'] = torch.exp(log_prob_s[:, :-1]).detach()

            if not self.training:
                aux[expert.name + '_edit_distance'] = torch.Tensor([self.get_edit_distance(message, sender_output)])
                aux['msg_lengths'] = find_lengths(sender_output.cpu()).float().mean()
            else:
                aux[expert.name + '_edit_distance'] = None
                aux['msg_lengths'] = None

        interaction = Interaction(
            sender_input=sender_input,
            receiver_input=None,
            labels=None,
            aux_input=None,
            message=None,
            receiver_output=None,
            receiver_sample=None,
            message_length=None,
            aux=aux,
        ).to('cpu')

        return losses, accs, entropies, interaction


def validate(val_dataloader, imitator, experts, val_losses, val_accs, val_edits, val_lengths, t, opts):
    val_iterator = iter(val_dataloader)
    imitator.eval()
    val_data = next(val_iterator)
    val_loss, val_acc, val_entropy, val_interaction = imitator.score(val_data, experts, print_outputs=True)

    del val_iterator
    for i in range(len(experts)):
        expert_name = experts[i].name
        val_losses[i].append(val_loss[i])
        val_accs[i].append(val_acc[i])
        val_edits[i].append(val_interaction.aux[expert_name + '_edit_distance'])

    val_lengths.append(val_interaction.aux['msg_lengths'])

    print_string = ['Expert: {}, Loss: {}, Acc: {} '.format(
        experts[i].name, val_losses[i][-1], val_accs[i][-1]) for i in range(len(experts))]
    print('VAL Epoch ' + str(t) + 'Overall Loss: {}'.format(
        sum([val_loss[-1] for val_loss in val_losses]) / len(experts)) + ' '.join(
        print_string))

    # save val interaction
    val_interaction = val_interaction.to('cpu')
    core.InteractionSaver.dump_interactions(val_interaction, 'validation', t, 0,
                                            opts.checkpoint_dir + '/randomseed_{}'.format(opts.random_seed))


def main(opts):
    # Get data
    train_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    train_data = one_hotify(train_data, opts.n_attributes, opts.n_values)
    train_data_extended = train_data * opts.data_scaler
    train_batch_size = min(opts.batch_size, len(train_data_extended))
    train_dataloader = DataLoader(train_data_extended, batch_size=train_batch_size)
    val_dataloader = DataLoader(train_data, batch_size=train_batch_size)

    # Get experts
    experts = get_experts(opts)

    # Get sender architecture
    sender = Sender(n_inputs=opts.n_attributes * opts.n_values, n_hidden=opts.hidden)

    sender = RnnSenderReinforceImitator(
        agent=sender,
        vocab_size=opts.vocab_size if not opts.variable_message_length else opts.vocab_size + 1, # 0 is reserved for EOS
        embed_dim=opts.sender_emb,
        hidden_size=opts.hidden,
        max_len=opts.max_len,
        cell=opts.sender_cell,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imitator = Imitator(sender, opts, device, len(experts))

    optimizer = torch.optim.Adam(imitator.parameters(), lr=opts.lr)

    losses = [[] for _ in experts]
    accs = [[] for _ in experts]
    entropies = [[] for _ in experts]
    val_losses = [[] for _ in experts]
    val_accs = [[] for _ in experts]
    val_edit_dists = [[] for _ in experts]
    val_msg_lengths = []

    for t in range(opts.n_epochs):
        iterator = iter(train_dataloader)
        epoch_losses = [[] for _ in experts]
        epoch_accs = [[] for _ in experts]
        epoch_entropies = [[] for _ in experts]

        if t % 10 == 0:
            validate(val_dataloader, imitator, experts, val_losses, val_accs, val_edit_dists, val_msg_lengths, t, opts)

        imitator.train()
        try:
            train_interactions = []
            while True:
                batch_data = next(iterator)
                optimizer.zero_grad()
                loss, acc, entropy, interaction = imitator.score(batch_data, experts, print_outputs=False)
                train_interactions.append(interaction)

                for i in range(len(loss)):
                    epoch_losses[i].append(loss[i])
                    epoch_accs[i].append(acc[i])
                    epoch_entropies[i].append(entropy[i])

                composite_loss = sum(loss) / len(loss)
                composite_loss.backward()
                optimizer.step()
        except StopIteration:
            pass
        finally:
            del iterator

            train_interaction = Interaction.from_iterable(train_interactions).to('cpu')
            core.InteractionSaver.dump_interactions(train_interaction,
                                               'train', t, 0,
                                               opts.checkpoint_dir + '/randomseed_{}'.format(opts.random_seed))

            for i in range(len(epoch_losses)):
                losses[i].append(sum(epoch_losses[i])/len(epoch_losses[i]))
                accs[i].append(sum(epoch_accs[i]) / len(epoch_accs[i]))
                entropies[i].append(sum(epoch_entropies[i]) / len(epoch_entropies[i]))
            print_string = ['Expert: {}, Loss: {}, Acc: {}'.format(
                experts[i].name, losses[i][-1], accs[i][-1]) for i in range(len(epoch_losses))]
            print('Epoch ' + str(t) + ' Overall Loss: {}'.format(sum([loss[-1] for loss in losses]) / len(experts)) + ' '.join(print_string))

    # Analysis
    optim = 'reinforce' if opts.reinforce else 'backprop'

    def plot_graph(time_series, titl, savepath):
        for i, loss in enumerate(time_series):
            plt.plot(range(0, opts.n_epochs, opts.n_epochs // len(loss)), loss, label=experts[i].name)
            plt.xlabel('Epoch')
            plt.ylabel(titl)
        plt.legend()
        plt.savefig('{}_{}_'.format(optim, '_'.join(opts.experts)) + savepath)
        plt.close()

    plot_graph(losses, "Training Loss", 'loss.png')
    plot_graph(val_losses, "Validation Loss", 'val_loss.png')
    plot_graph(val_accs, "Expert policy weights", 'exp_weights.png')
    plot_graph(accs, "Training Accuracy", "acc.png")
    plot_graph(val_accs, "Validation Accuracy", 'val_acc.png')
    plot_graph(val_edit_dists, "Levenstein Distance", 'edit_dist.png')
    plot_graph(entropies, "Imitator Entropy", "entropy.png")
    plot_graph([val_msg_lengths], "Message length", 'message_length.png')


if __name__=="__main__":
    import sys
    args = sys.argv[1:]
    args = get_params(args)
    args.experts = [''.join(expert) for expert in args.experts] # hack to resolve a command line bug
    args.expert_lengths = [''.join(expert) for expert in args.expert_lengths]

    print(args)
    main(args)