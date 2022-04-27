# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import egg.core as core
from egg.zoo.addition.archs import Game, Receiver, Sender, PlusOneWrapper
from egg.zoo.addition.losses import get_loss


def define_agents(opts):
    receiver = Receiver(n_hidden=opts.hidden, n_outputs=opts.input_size * 2 - 1)
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_emb,
        opts.hidden,
        cell=opts.receiver_cell,
    )

    sender = Sender(n_inputs= 2 * opts.input_size, n_hidden=opts.hidden)
    sender = core.RnnSenderReinforce(
        agent=sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_emb,
        hidden_size=opts.hidden,
        max_len=opts.max_len,
        cell=opts.sender_cell,
    )
    sender = PlusOneWrapper(sender)

    return sender, receiver


def build_optimizer_and_scheduler(
    game: nn.Module, lr: float
) -> Tuple[
    torch.optim.Optimizer, Optional[Any]
]:  # some pytorch schedulers are child classes of object
    optimizer = torch.optim.Adam(game.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return optimizer


def build_game(opts: argparse.Namespace) -> nn.Module:
    loss = get_loss(opts)
    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    sender, receiver = define_agents(opts)
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
    )
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
