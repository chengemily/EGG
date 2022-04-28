# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn

from egg.core.interaction import Interaction


class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.fc = nn.Linear(n_inputs, n_hidden)

    def forward(
        self,
        sender_input: torch.Tensor,
        aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        return self.fc(sender_input)


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_outputs):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        receiver_input: torch.Tensor = None,
        aux_input: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        return self.fc(x)


class Game(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, Dict[str, Any]],
        ],
    ):
        super(Game, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Interaction]:
        pass
