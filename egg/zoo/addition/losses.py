# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn.functional as F


def get_loss(opts) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, Dict[str, Any]],
]:
    return Loss(opts.input_size)


class Loss:
    def __init__(self, input_size):
        self.sender_input_size = input_size
        self.rcvr_output_size = input_size * 2 - 1

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_input: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(batch_size, self.rcvr_output_size)
        labels = labels.view(batch_size, self.rcvr_output_size)

        acc = (
                receiver_output.argmax(dim=-1) == labels.argmax(dim=-1)
            ).float()
        print(acc.shape)
        print('ACC: ', acc)
        loss = F.cross_entropy(receiver_output, labels.argmax(dim=-1), reduction='mean')
        print('LOSS: ', loss)
        input()
        return loss, {"acc": acc}
