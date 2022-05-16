# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from typing import Tuple, List, Optional, Dict
import random


class DataScaler(Dataset):
    def __init__(self, dataset: Dataset, scaling_factor: int):
        self.dataset = dataset
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.dataset) * self.scaling_factor

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class SummationDataset(Dataset):
    def __init__(self, n_inputs, weights=None, frame=None):
        self.n_inputs = n_inputs

        if frame is not None:
            self.frame = frame
        else:
            self.frame = []
            self.weights = torch.ones(self.n_inputs * 2 - 1) if weights is None else weights
            for i in range(n_inputs):
                for j in range(n_inputs):
                    z = torch.zeros(2 * n_inputs)
                    z[i] = 1
                    z[n_inputs + j] = 1
                    label = torch.tensor(i + j)

                    for _ in range(int(self.weights[i+j])):
                        self.frame.append((z, label))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


def get_dataloaders(n_inputs: int, training_density: float, train_batch_size: int, scaling_factor: int=3) -> \
        Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:

    # Get datasets
    train_ds, val_ds, holdout_ds = dataset_split(n_inputs, training_density)
    train_ds_scaled = DataScaler(train_ds, scaling_factor=scaling_factor)

    if holdout_ds is not None:
        comb_ds, perm_ds = get_combination_and_permutation_holdouts(holdout_ds, n_inputs)
        holdout_loaders = {
            'uniform': DataLoader(
                            holdout_ds, batch_size=len(holdout_ds), shuffle=False, num_workers=1,
                        )
        }
        if len(perm_ds):
            perm_loader = DataLoader(
                perm_ds, batch_size=len(perm_ds), shuffle=False, num_workers=1,
            )
            holdout_loaders['permutation'] = perm_loader
        if len(comb_ds):
            comb_loader = DataLoader(
                comb_ds, batch_size=len(comb_ds), shuffle=False, num_workers=1,
            )
            holdout_loaders['combination'] = comb_loader
    else:
        holdout_loaders = None

    # Build weighted sampler for training set
    class_frequencies, _ = np.histogram([int(label) for _, label in train_ds],
                                     bins=[i - 0.5 for i in range(2 * n_inputs)])
    weights = [1 / class_frequencies[int(label)] for _, label in train_ds]
    weights.extend(weights * (scaling_factor - 1))
    weights = torch.tensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Build dataloaders
    train_loader = DataLoader(
        train_ds_scaled, batch_size=train_batch_size,
        sampler=sampler,
        num_workers=1
    )

    test_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, num_workers=1)

    return train_loader, test_loader, holdout_loaders


def dataset_split(n_inputs: int, training_density: float=0.7) -> \
        Tuple[
            SummationDataset, SummationDataset, Optional[SummationDataset]
        ]:
    """
    Splits data into a training, validation (same as training data), and a holdout set.
    :param training_density: % of input data size[n_input x n_input] in training set.
    """
    holdout_set = None

    if training_density < 1:
        full_dataset = SummationDataset(n_inputs)
        training_length = int(training_density * len(full_dataset))

        training_set, holdout_set = torch.utils.data.random_split(
            full_dataset,
            [training_length, len(full_dataset) - training_length],
            generator=torch.Generator().manual_seed(7)
        )
        holdout_labels = set([int(label) for _, label in list(holdout_set)])
        training_labels = set([int(label) for _, label in list(training_set)])

        assert holdout_labels.issubset(training_labels)
    else:
        training_set = SummationDataset(n_inputs)

    return training_set, training_set, holdout_set


def get_combination_and_permutation_holdouts(this_set: Dataset, n_inputs: int) -> \
        Tuple[SummationDataset, SummationDataset]:
    """
    Compares self to the frame of another SummationDataset. Returns a new SummationDataset consisting of all the
    datapoints in self where any permutation of input is not in other_set.
    """
    this_set_data = list(map(lambda x: x[0].tolist(), this_set))
    combination_frame = []
    permutation_frame = []

    for x, label in set(this_set):
        reverse_x = torch.cat([x[n_inputs:], x[:n_inputs]]).tolist()

        if (reverse_x in this_set_data):
            combination_frame.append((x, label))
        else:
            permutation_frame.append((x, label))

    return SummationDataset(n_inputs, frame=combination_frame), \
           SummationDataset(n_inputs, frame=permutation_frame)

