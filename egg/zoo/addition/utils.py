# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_other_opts(parser):
    group = parser.add_argument_group("other")
    group.add_argument("--dummy", type=float, default=1.0, help="dumy option")


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument(
        "--input_size",
        default=20,
        help="Input size N is numbers 0...N-1",
    )
    parser.add_argument(
        "--sender_cell",
        default='gru',
        choices=['lstm', 'rnn', 'gru'],
        help='Type of sender cell'
    )
    parser.add_argument(
        "--receiver_cell",
        default='gru',
        type=str,
        choices=['lstm', 'rnn', 'gru'],
        help='Type of receiver cell'
    )
    parser.add_argument(
        "--sender_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=100,
        help='Size of hidden layers'
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default='mean',
        choices=['no', 'mean', 'builtin'],
        help='Whether to use a baseline in policy loss'
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )
    parser.add_argument(
        "--training_density",
        type=float,
        default=0.7,
        help="Proportion of full dataset that is the training set",
    )

    # get_other_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
