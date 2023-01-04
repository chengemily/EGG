# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import time
from csv import DictWriter
from typing import Dict, Iterable
from collections import defaultdict

import torch
from scipy import spatial
from scipy.stats import spearmanr

import egg.core as core
from egg.core.batch import Batch

from egg.core import Callback, Interaction, InteractionSaver
from egg.core.early_stopping import EarlyStopper
from egg.core.language_analysis import TopographicSimilarity, Disent

from egg.zoo.imitation_learning.bc_trainer import BCTrainer
from egg.zoo.imitation_learning.behavioural_cloning import bc_agents_setup, train_bc
from egg.zoo.imitation_learning.archs import define_agents
from egg.zoo.imitation_learning.util import entropy, mutual_info
# from egg.zoo.imitation_learning.mixed_game import MultitaskGame


def editdistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def ask_sender(n_attributes, n_values, dataset, sender, device):
    attributes = []
    strings = []
    meanings = []

    for i in range(len(dataset)):
        meaning = dataset[i]

        attribute = meaning.view(n_attributes, n_values).argmax(dim=-1)
        attributes.append(attribute)
        meanings.append(meaning.to(device))

        with torch.no_grad():
            string, *other = sender(meaning.unsqueeze(0).to(device))
        strings.append(string.squeeze(0))

    attributes = torch.stack(attributes, dim=0)
    strings = torch.stack(strings, dim=0)
    meanings = torch.stack(meanings, dim=0)

    return attributes, strings, meanings


<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def information_gap_position(n_attributes, n_values, dataset, sender, device):
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )
    return information_gap_representation(attributes, strings)


def histogram(strings, vocab_size):
    batch_size = strings.size(0)

    histogram = torch.zeros(batch_size, vocab_size, device=strings.device)

    for v in range(vocab_size):
        histogram[:, v] = strings.eq(v).sum(dim=-1)

    return histogram


def information_gap_vocab(n_attributes, n_values, dataset, sender, device, vocab_size):
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )

    histograms = histogram(strings, vocab_size)
    return information_gap_representation(attributes, histograms[:, 1:])


def edit_dist(_list):
    distances = []
    count = 0
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i + 1 :]):
            count += 1
            # Normalized edit distance (same in our case as length is fixed)
            distances.append(editdistance(el1, el2) / len(el1))
    return distances


def cosine_dist(_list):
    distances = []
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i + 1 :]):
            distances.append(spatial.distance.cosine(el1, el2))
    return distances


def topographic_similarity(n_attributes, n_values, dataset, sender, device):
    _attributes, strings, meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )
    list_string = []
    for s in strings:
        list_string.append([x.item() for x in s])
    distance_messages = edit_dist(list_string)
    distance_inputs = cosine_dist(meanings.cpu().numpy())

    corr = spearmanr(distance_messages, distance_inputs).correlation
    return corr


def context_independence(n_attributes, n_values, dataset, sender, device) -> float:
    # see https://github.com/tomekkorbak/measuring-non-trivial-compositionality/blob/master/metrics/context_independence.py
    num_concepts = n_values * n_attributes
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )

    character_set = set(c.item() for message in strings for c in message)
    character_set.discard(1) # 1 is just the END token
    vocab = {char: idx for idx, char in enumerate(character_set)} # create a dictionary
    concept_set = set([(att, val.item()) for object in attributes for att, val in enumerate(object)]) # concept = (att,val) tuple
    concepts = {concept: idx for idx, concept in enumerate(concept_set)}
    concept_symbol_matrix = compute_concept_symbol_matrix(attributes, strings, num_concepts, vocab, concepts)
    v_cs = torch.argmax(concept_symbol_matrix, dim=1)

    context_independence_scores = torch.zeros(len(concept_set))
    for concept_idx in range(len(concept_set)):
        v_c = v_cs[concept_idx]
        p_c_vc = concept_symbol_matrix[concept_idx, v_c] / torch.sum(concept_symbol_matrix[:, v_c], dim=0)
        p_vc_c = concept_symbol_matrix[concept_idx, v_c] / torch.sum(concept_symbol_matrix[concept_idx, :], dim=0)
        context_independence_scores[concept_idx] = p_vc_c * p_c_vc
    return torch.mean(context_independence_scores, dim=0).item()


def compute_concept_symbol_matrix(
        objects,
        messages,
        num_concepts,
        vocab: Dict[int, int],
        concepts: Dict[tuple, int],
        epsilon: float = 10e-8
    ):
    # Builds frequency matrix
    concept_to_message = defaultdict(list)
    for i, message in enumerate(messages):
        object = objects[i]
        for att, value in enumerate(object):
            concept = (att, value.item())
            concept_to_message[concept] += list(message[:-1])  # get rid of 1 EOS

    concept_symbol_matrix = torch.full((num_concepts, len(vocab)), epsilon)
    for concept, symbols in concept_to_message.items():
        for symbol in symbols:
            if symbol.item() != 1:  # don't count EOS
                concept_symbol_matrix[concepts[concept], vocab[symbol.item()]] += 1
    return concept_symbol_matrix


class CompoEvaluator(core.Callback):
    def __init__(self, dataset, device, n_attributes, n_values, vocab_size, is_distributed, is_population=False, epochs=[]):
        self.dataset = dataset
        self.device = device
        self.is_distributed = is_distributed
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.epoch = 0
        self.vocab_size = vocab_size
        self.epochs_to_save = epochs
        self.stats = {}
        self.population_based = is_population

    def dump_stats(self):
        game = self.trainer.game
        game.eval()

        if not self.population_based:
            # Makes one sender compatible with population of senders (which is iterated over and aggregated)
            senders = [game.module.sender] if self.is_distributed else [game.sender]
        else:
            senders = game.module.senders if self.is_distributed else game.senders

        stats = []
        for sender in senders:
            attributes, messages, meanings = ask_sender(
                self.n_attributes, self.n_values, self.dataset, sender, self.device
            )
            positional_disent = Disent.posdis(attributes, messages)
            bos_disent = Disent.bosdis(attributes, messages, self.vocab_size)
            topo_sim = TopographicSimilarity.compute_topsim(meanings.cpu(), messages.cpu())
            context_ind = context_independence(self.n_attributes, self.n_values, attributes, messages)

            output = dict(
                epoch=self.epoch,
                positional_disent=positional_disent,
                bag_of_symbol_disent=bos_disent,
                topographic_sim=topo_sim,
                context_independence=context_ind
            )
            stats.append(output)

        self.stats = dict(
            epoch=[sender_stat['epoch'] for sender_stat in stats],
            positional_disent=[sender_stat['positional_disent'] for sender_stat in stats],
            bag_of_symbol_disent=[sender_stat['bag_of_symbol_disent'] for sender_stat in stats],
            topographic_sim=[sender_stat['topographic_sim'] for sender_stat in stats],
            context_independence=[sender_stat['context_independence'] for sender_stat in stats]
        )

        output_json = json.dumps(self.stats)
        print(output_json, flush=True)
        game.train()

    def on_train_end(self):
        self.dump_stats()

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        self.epoch += 1

        if self.epoch not in self.epochs_to_save:
            return

        self.dump_stats()

        # Save to interaction
        for k, v in self.stats.items():
            if k != 'epoch':
                logs.aux['compo_metrics/{}'.format(k)] = torch.Tensor(v)


class HoldoutEvaluator(core.Callback):
    def __init__(self, loaders_metrics, device, is_distributed, is_population, epochs=[]):


        self.loaders_metrics = loaders_metrics
        self.device = device
        self.epoch = 0
        self.epochs = epochs
        self.is_distributed = is_distributed
        self.is_population = is_population
        self.results = {}

    def evaluate(self):
        game = self.trainer.game
        game_skip_bc = game.skip_bc
        game.eval()
        old_loss = game.loss if not self.is_distributed else game.module.loss

        for loader_name, loader, metric in self.loaders_metrics:
            acc_or, acc = 0.0, 0.0
            n_batches = 0
            if self.is_distributed:
                game.module.loss = metric
            else:
                game.loss = metric

            for batch in loader:
                n_batches += 1
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                
		with torch.no_grad():
                    _, interaction = game(*batch)
                
		acc += interaction.aux["acc"].mean().item()

                acc_or += interaction.aux["acc_or"].mean().item()
            self.results[loader_name] = {
                "acc": acc / n_batches,
                "acc_or": acc_or / n_batches,
            }

        self.results["epoch"] = self.epoch
        output_json = json.dumps(self.results)
        print(output_json, flush=True)

        if self.is_distributed:
            game.module.loss = old_loss
        else:
            game.loss = old_loss
        game.train()
        game.skip_bc = game_skip_bc

    def evaluate_population(self):
        game = self.trainer.game
        game.eval()
        old_loss = game.loss if not self.is_distributed else game.module.loss
        number_of_pairs = game.number_of_pairs if not self.is_distributed else game.module.number_of_pairs

        for loader_name, loader, metric in self.loaders_metrics:
            acc_or, acc = [], []
            n_batches = 0
            if self.is_distributed:
                game.module.loss = metric
            else:
                game.loss = metric

            for batch in loader:
                n_batches += 1
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                # print('HERE')
                _, interaction = game(*batch)

                if loader_name == 'generalization hold out':
                    acc.append(interaction.aux['acc'])
                    acc_or.append(interaction.aux['acc_or'])
                elif loader_name == 'uniform holdout':
                    # transform into # senders x batch size x .. ...
                    batch_size = interaction.aux['acc'].size(0)
                    n_attributes = interaction.aux['acc_or'].size(1)

                    acc.append(interaction.aux["acc"].view(number_of_pairs, batch_size // number_of_pairs).mean(dim=-1))
                    acc_or.append(interaction.aux["acc_or"].view(
                        number_of_pairs, batch_size // number_of_pairs, n_attributes).mean(dim=[1, 2]))
            self.results[loader_name] = {
                'population_acc': sum([pair.mean() for pair in acc]).item() / n_batches,
                'population_acc_or': sum([pair.mean() for pair in acc_or]).item() / n_batches,
                "acc": (sum(acc) / n_batches).tolist(),
                "acc_or": (sum(acc_or) / n_batches).tolist(),
            }

        self.results["epoch"] = [self.epoch]
        output_json = json.dumps(self.results)
        print(output_json, flush=True)

        if self.is_distributed:
            game.module.loss = old_loss
        else:
            game.loss = old_loss
        game.train()

    def on_train_end(self):
        self.evaluate() if not self.is_population else self.evaluate_population()
        game.loss = old_loss
        game.train()


    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        self.epoch += 1

        if self.epoch not in self.epochs: return
        self.evaluate() if not self.is_population else self.evaluate_population()
        
	# Save to Interaction
        for loader_name in self.results:
            if loader_name == 'epoch':
                continue
            for k, v in self.results[loader_name].items():
                if type(v) != list: v = [v]
                logs.aux['{}/{}'.format(loader_name, k)] = torch.Tensor(v)
        

class BehaviouralCloning(core.Callback):
    def __init__(self, opts, bc_opts, optimizer, kick: str, freq: int=0, sender_rcvr_imitation_reinforce_weight: float=0.0):
        self.expert_optimizer = optimizer
        self.bc_opts = bc_opts
        self.opts = opts
        self.device = opts.device
        # whether a REINFORCE loss from receiver imitation should kick back to speaker
        self.sender_rcvr_imitation_reinforce_weight = sender_rcvr_imitation_reinforce_weight

        self.kick = kick # imitation, none, or fixed
        self.ablation = opts.ablation # all, receiver_only, or sender_only
        self.epoch = 0
        self.freq = freq
        self.results = {}
        self.bc_trainer = BCTrainer(self.bc_opts)

    def on_train_begin(self, trainer_instance: "Trainer"):
        self.trainer = trainer_instance
        self.receiver, self.sender = self.trainer.game.receiver, self.trainer.game.sender

    
    def train_bc(self, logs):
        bc_speaker, bc_receiver = bc_agents_setup(self.opts, self.device, *define_agents(self.opts))
        bc_optimizer_r = torch.optim.Adam(bc_receiver.parameters(), lr=self.opts.lr)
        bc_optimizer_s = torch.optim.Adam(bc_speaker.parameters(), lr=self.opts.lr)

        # Train bc agents until convergence, logging all the while.
        if self.kick == 'imitation':
            self.expert_optimizer.zero_grad()

        s_loss, r_loss, t, last_s_acc, last_r_acc, cumu_s_acc, cumu_r_acc = train_bc(
            self.bc_opts,
            bc_speaker, bc_receiver,
            bc_optimizer_s, bc_optimizer_r,
            self.trainer,
            imitation=(self.kick == 'imitation'),
            sender_aware_weight=self.sender_rcvr_imitation_reinforce_weight
        )

        results = dict(
            epoch=self.epoch,
            bc_s_loss=s_loss.item(),
            bc_r_loss=r_loss.item(),
            imitation_s_acc=last_s_acc.item(),
            imitation_r_acc=last_r_acc.item(),
            sample_complexity=t,
        )
        self.results = results

        if self.kick == 'random':
            sender_grads, rcvr_grads = load_gradients(self.opts)
            with torch.no_grad():
                for param in game.sender.parameters():
                    param.add_(
                        torch.randn(param.size(), device=device) * sender_grads[curr_epoch] * torch.sqrt(torch.pi / 2))
                for param in game.receiver.parameters():
                    param.add_(
                        torch.randn(param.size(), device=device) * rcvr_grads[curr_epoch] * torch.sqrt(torch.pi / 2))

        if self.kick == 'imitation':
            self.expert_optimizer.step()

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        # print('in bc callback')
        self.epoch += 1

        if self.freq > 0 and (self.epoch % self.freq == 0 or self.epoch == 1):
            self.train_bc(logs)
        # Save to Interaction
        for k, v in self.results.items():
            logs.aux['imitation/{}'.format(k)] = torch.Tensor([v])  # backwards compatibility w/ core.

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        self.epoch += 1

        if self.freq <= 0 or self.epoch % self.freq != 0:
            return

        self.train_bc(logs)


class BehaviouralCloningConvergence(EarlyStopper):
    """
    Implements early stopping logic that stops training when a threshold on a metric
    is achieved.
    """

    def __init__(
        self, opts, bc_opts, optimizer,
            kick: str, threshold: float=1.1, conv_epsilon:float=0.0,
            interactions_dir: str='./interactions',
            field_name: str = "acc", validation: bool = False,
            sender_rcvr_imitation_reinforce_weight: float=0.0
    ) -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name `field_name`)
        :param field_name: the name of the metric return by loss function which should be evaluated against stopping
            criterion (default: "acc")
        :param validation: whether the statistics on the validation (or training, if False) data should be checked
        """
        super(BehaviouralCloningConvergence, self).__init__(validation)
        self.opts = opts
        self.bc_opts = bc_opts
        self.kick = kick
        self.device = opts.device
        self.expert_optimizer = optimizer
        self.threshold = threshold
        self.conv_epsilon = conv_epsilon
        self.ablation = opts.ablation # all, receiver_only, or sender_only
        self.field_name = field_name
        self.sender_rcvr_imitation_reinforce_weight = sender_rcvr_imitation_reinforce_weight
        self.interactions_dir = interactions_dir

    def on_train_begin(self, trainer_instance: "Trainer"):
        self.trainer = trainer_instance
        self.bc_trainer = BCTrainer(self.bc_opts,
                                    self.trainer,
                                    self.ablation,
                                    self.bc_opts.n_epochs_bc,
                                    imitation=(self.kick == 'imitation'),
                                    sender_aware_weight=self.sender_rcvr_imitation_reinforce_weight)


    def train_bc(self, logs):
        bc_speaker, bc_receiver = bc_agents_setup(self.opts, self.device, *define_agents(self.opts))
        bc_optimizer_r = torch.optim.Adam(bc_receiver.parameters(), lr=self.opts.lr)
        bc_optimizer_s = torch.optim.Adam(bc_speaker.parameters(), lr=self.opts.lr)

        # Train bc agents until convergence, logging all the while.
        if self.kick == 'imitation':
            self.expert_optimizer.zero_grad()

        s_loss, r_loss, \
        s_policy_loss, r_policy_loss, \
        s_t, r_t, \
        last_s_acc, last_r_acc, cumu_s_acc, cumu_r_acc = self.bc_trainer.train(
            bc_speaker,
            bc_receiver,
            bc_optimizer_r,
            bc_optimizer_s
        )

        if self.kick == 'none':
            s_policy_loss, r_policy_loss = s_policy_loss.detach(), r_policy_loss.detach()
            s_loss, r_loss = s_loss.detach(), r_loss.detach()

        results = dict(
            epoch=self.epoch,
            bc_s_loss=s_loss.item(),
            bc_r_loss=r_loss.item(),
            bc_s_policy_loss=s_policy_loss.item(),
            bc_r_policy_loss=r_policy_loss.item(),
            imitation_s_acc=last_s_acc.item(),
            imitation_r_acc=last_r_acc.item(),
            sol_s=cumu_s_acc.item(),
            sol_r=cumu_r_acc.item(),
            sample_complexity=max(s_t, r_t),
            sender_sample_complexity=s_t,
            receiver_sample_complexity=r_t
            imitation_s_acc=last_s_acc.item(),
            imitation_r_acc=last_r_acc.item(),
            sample_complexity=t,
        )
        self.results = results

        if self.kick == 'random':
            sender_grads, rcvr_grads = load_gradients(self.opts)
            with torch.no_grad():
                for param in game.sender.parameters():
                    param.add_(
                        torch.randn(param.size(), device=device) * sender_grads[curr_epoch] * torch.sqrt(torch.pi / 2))
                for param in game.receiver.parameters():
                    param.add_(
                        torch.randn(param.size(), device=device) * rcvr_grads[curr_epoch] * torch.sqrt(torch.pi / 2))

        if self.kick == 'imitation':
            self.expert_optimizer.step()

        # Save to Interaction
        for k, v in self.results.items():
            logs.aux['imitation/{}'.format(k)] = torch.Tensor([v])  # backwards compatibility w/ core.

        # Dump interaction
        InteractionSaver.dump_interactions(
            logs,
            'train',
            self.epoch,
            self.trainer.distributed_context.rank,
            self.interactions_dir
        )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int) -> None:
        if self.validation:
            return
        self.epoch = epoch
        self.train_stats.append((loss, logs))
        if self.should_stop():
            self.trainer.game.receiver.deterministic_mode()
            self.train_bc(logs)
            self.trainer.game.receiver.reinforce_mode()

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int) -> None:
        if not self.validation:
            return
        self.validation_stats.append((loss, logs))
        if self.should_stop():
            pass

    def should_stop(self) -> bool:
        if self.validation:
            assert (
                self.validation_stats
            ), "Validation data must be provided for early stooping to work"
            loss, last_epoch_interactions = self.validation_stats[-1]
        else:
            assert (
                self.train_stats
            ), "Training data must be provided for early stooping to work"
            loss, last_epoch_interactions = self.train_stats[-1]

        metric_mean = last_epoch_interactions.aux[self.field_name].mean()

        return metric_mean >= self.threshold or \
               max(self.trainer.last_grad_receiver, self.trainer.last_grad_sender) < self.conv_epsilon
