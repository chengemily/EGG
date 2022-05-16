# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class Receiver(nn.Module):
    def __init__(self, n_output, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, int((n_hidden + n_output) / 2))
        self.tanh = nn.Tanh()
        self.output = nn.Linear(int((n_hidden + n_output) / 2), n_output)

    def forward(self, x, _input, _aux_input):
        return self.output(self.tanh(self.fc1(x)))


# The Sender class implements the core Sender agent common to both games: it gets the input target vector and produces a hidden layer
# that will initialize the message producing RNN
class Sender(nn.Module):
    def __init__(self, n_hidden, n_inputs):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.tanh = nn.Tanh()

    def forward(self, x, _aux_input):
        return self.tanh(self.fc1(x))
