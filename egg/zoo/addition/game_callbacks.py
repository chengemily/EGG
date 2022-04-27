# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import argparse

from egg.core import Callback, ConsoleLogger, EarlyStopperAccuracy, TensorboardLogger


def get_callbacks(opts, is_distributed: bool = False) -> List[Callback]:
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        # TensorboardLogger(),
        EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)
    ]

    return callbacks
