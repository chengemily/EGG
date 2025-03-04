# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from typing import List

import torch

import egg.core as core
from egg.zoo.addition.data import get_dataloaders
from egg.zoo.addition.game_callbacks import get_callbacks
from egg.zoo.addition.games import build_game, build_optimizer_and_scheduler
from egg.zoo.addition.utils import get_opts


def main(params: List[str]) -> None:
    begin = datetime.now() + timedelta(hours=6)
    print(f"| STARTED JOB at {begin}...")

    opts = get_opts(params=params)
    print(f"{opts}\n")
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader, val_loader, holdout_loader = get_dataloaders(opts)

    game = build_game(opts)
    optimizer = build_optimizer_and_scheduler(game, opts.lr)
    callbacks = get_callbacks(opts)
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)
    early_stopper = callbacks[1]
    last_interaction = early_stopper.validation_stats[-1][1]
    print(last_interaction)
    end = datetime.now() + timedelta(hours=6)  # Using CET timezone

    print(f"| FINISHED JOB at {end}. It took {end - begin}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
