# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

class ScaledDataset:
    def __init__(self, examples, scaling_factor=1):
        self.examples = examples
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.examples) * self.scaling_factor

    def __getitem__(self, k):
        k = k % len(self.examples)
        return self.examples[k]


def get_dataloaders(opts) -> Tuple[Iterable[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
], Iterable[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
]]:
    "Returning an iterator for tuple(sender_input, labels, receiver_input)."
    full_data = enumerate_dataset(opts.input_size)
    len_train = int(opts.training_density * len(full_data))
    train_set, holdout_set = torch.utils.data.random_split(full_data,
                                                           [len_train, len(full_data) - len_train]
                                                           )
    validation_set = train_set

    train_set = ScaledDataset(train_set, opts.data_scaler)

    train_loader, validation_loader, holdout_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True), \
                                             DataLoader(validation_set, batch_size=len(validation_set)), \
                                             DataLoader(holdout_set, batch_size=opts.batch_size)

    return train_loader, validation_loader, holdout_loader


def enumerate_dataset(input_size):
    data = []
    labels = []

    for i in range(input_size):
        for j in range(input_size):
            inp = torch.zeros(2 * input_size)
            inp[i] = 1.0
            inp[input_size + j] = 1.0

            label = torch.zeros(2 * input_size - 1)
            label[i + j] = 1.0

            data.append(inp)
            labels.append(label)

    data_tuples = [(data[i], labels[i]) for i in range(len(data))]
    return data_tuples



