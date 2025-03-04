# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from csv import DictWriter
from typing import Dict
from collections import defaultdict

import torch
from scipy import spatial
from scipy.stats import spearmanr

import egg.core as core
from egg.core.batch import Batch
from egg.zoo.language_bottleneck.intervention import entropy, mutual_info

try:
    import editdistance  # package to install https://pypi.org/project/editdistance/0.3.1/
except ImportError:
    print(
        "Please install editdistance package: `pip install editdistance`. "
        "It is used for calculating topographic similarity."
    )


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
            distances.append(editdistance.eval(el1, el2) / len(el1))
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
            concept_to_message[concept] += list(message[:-1]) # get rid of 1 EOS

    concept_symbol_matrix = torch.full((num_concepts, len(vocab)), epsilon)
    for concept, symbols in concept_to_message.items():
        for symbol in symbols:
            if symbol.item() != 1: # don't count EOS
                concept_symbol_matrix[concepts[concept], vocab[symbol.item()]] += 1
    return concept_symbol_matrix


class Metrics(core.Callback):
    def __init__(self, dataset, device, n_attributes, n_values, vocab_size, freq=1):
        self.dataset = dataset
        self.device = device
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.epoch = 0
        self.vocab_size = vocab_size
        self.freq = freq
        self.stats = {}

    def dump_stats(self):
        game = self.trainer.game
        game.eval()

        positional_disent = information_gap_position(
            self.n_attributes, self.n_values, self.dataset, game.sender, self.device
        )
        bos_disent = information_gap_vocab(
            self.n_attributes,
            self.n_values,
            self.dataset,
            game.sender,
            self.device,
            self.vocab_size,
        )
        topo_sim = topographic_similarity(
            self.n_attributes, self.n_values, self.dataset, game.sender, self.device
        )
        context_ind = context_independence(self.n_attributes, self.n_values, self.dataset, game.sender, self.device)

        output = dict(
            epoch=self.epoch,
            positional_disent=positional_disent,
            bag_of_symbol_disent=bos_disent,
            topographic_sim=topo_sim,
            context_independence=context_ind
        )
        self.stats = output
        output_json = json.dumps(output)
        print(output_json, flush=True)
        game.train()

    def on_train_end(self):
        self.dump_stats()

    def on_epoch_end(self, *stuff):
        self.epoch += 1

        if self.freq <= 0 or self.epoch % self.freq != 0:
            return

        self.dump_stats()


class Evaluator(core.Callback):
    def __init__(self, loaders_metrics, device, freq=1):
        self.loaders_metrics = loaders_metrics
        self.device = device
        self.epoch = 0
        self.freq = freq
        self.results = {}

    def evaluate(self):
        game = self.trainer.game
        game.eval()
        old_loss = game.loss

        for loader_name, loader, metric in self.loaders_metrics:

            acc_or, acc = 0.0, 0.0
            n_batches = 0
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

        game.loss = old_loss
        game.train()

    def on_train_end(self):
        self.evaluate()

    def on_epoch_end(self, *stuff):
        self.epoch += 1

        if self.freq <= 0 or self.epoch % self.freq != 0:
            return
        self.evaluate()
