# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

import torch

import egg.core as core
from egg.core.batch import Batch
from egg.core.interaction import Interaction
from egg.core.util import get_summary_writer
from egg.zoo.addition_game.data_readers import SummationDataset


class HoldoutEvaluator(core.Callback):
    def __init__(self, holdout_data, device, interaction_saver):
        self.loaders = holdout_data
        self.device = device
        self.interaction_saver = interaction_saver
        self.results = {}
        self.interaction = {}
        self.epoch = 0

    def evaluate(self):
        game = self.trainer.game
        game.eval()
        old_loss = game.loss

        for loader_name, loader in self.loaders.items():
            n_batches = 0
            acc = 0
            for batch in loader:
                n_batches += 1
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                with torch.no_grad():
                    _, interaction = game(*batch)
                acc += interaction.aux["acc"].mean().item()

            self.results[loader_name] = {
                "acc": acc / n_batches,
            }
            self.interaction[loader_name] = interaction

        self.results["epoch"] = self.epoch
        output_json = json.dumps(self.results)
        print(output_json, flush=True)

        game.loss = old_loss
        game.train()

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.epoch = epoch
        self.evaluate()
        rank = self.trainer.distributed_context.rank

        for loader_name, interaction in self.interaction.items():
            self.interaction_saver.dump_interactions(
                interaction, loader_name, epoch, rank, self.interaction_saver.checkpoint_dir
            )