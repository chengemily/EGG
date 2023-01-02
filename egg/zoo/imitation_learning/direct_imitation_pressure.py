# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import os
import copy
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core.early_stopping import *

from egg.zoo.imitation_learning.archs import (
    PlusOneWrapper,
    Receiver,
    Sender,
)

from egg.zoo.imitation_learning.callbacks import *
from egg.zoo.compo_vs_generalization.train import *

from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.behavioural_cloning import *
from egg.zoo.imitation_learning.util import *
from egg.zoo.imitation_learning.loss import DiffLoss
from egg.zoo.imitation_learning.game_direct_imitation import DirectImitationGame

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def imit_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, choices=['kl', 'cross_entropy'], default='cross_entropy')
    parser.add_argument('--sender_rcvr_imitation_reinforce_weight', type=float, default=0.0)
    parser.add_argument('--kick', type=str, choices=['none', 'imitation', 'random'], default='none')
    parser.add_argument('--ablation', type=str, choices=['all', 'sender_only', 'receiver_only', 'none'], default='all')
    parser.add_argument('--turn_taking', type=str, choices=['fixed', 'convergence', 'simultaneous'], default='fixed')
    parser.add_argument('--imitation_weight', type=float, default=0.0,
                        help="Weight of imitation reward compared to original task reward (weight=1)")
    parser.add_argument('--imitation_reinforce', type=boolean_string, default=False,
                        help="If True, treats imitation loss as (negative) reward from imitation task and trains with" +
                             "imitation_weight-ed REINFORCE.")
    parser.add_argument('--receiver_reinforce', type=int, default=0,
                        help="If True, trains receiver also by REINFORCE in the actual game.")
    parser.add_argument('--sample_pairs', type=boolean_string, default=False)
    parser.add_argument('--n_turns', type=int, default=50)
    parser.add_argument("--n_attributes",
                        type=int,
                        default=4, help="")
    parser.add_argument("--n_values",
                        type=int,
                        default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=1)
    parser.add_argument("--stats_freq", type=int, default=50)
    parser.add_argument(
        "--baseline", type=str, choices=["no", "mean", "builtin"], default="mean"
    )
    parser.add_argument(
        '--holdout_density', type=float, default=0.1
    )
    parser.add_argument('--hidden',
                        type=int,
                        default=50,
                        help='Size of hidden layer of both agents')
    parser.add_argument('--population_size',
                        type=int,
                        default=2,
                        help='Population size (must be an even number)')
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )
    parser.add_argument("--sender_cell", type=str, default="rnn")
    parser.add_argument("--receiver_cell", type=str, default="rnn")
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
        "--convergence_epsilon",
        type=float,
        default= 0.25,
        help="Stop training when gradient norm is less than epsilon."
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Set True if you want model to be saved",
    )

    args = core.init(arg_parser=parser, params=params)

    return args


def main(params, train_mode=True):
    import copy

    opts = imit_params(params)
    print('OPTS: ', opts)

    generalization_holdout_loader, uniform_holdout_loader, \
    full_data_loader, train_loader, validation_loader, \
    train, validation = load_data(opts)

    senders, receivers = [], []
    for i in range(opts.population_size):
        sender, receiver = define_agents(opts)
        senders.append(sender)
        receivers.append(receiver)

    loss = DiffLoss(opts.n_attributes, opts.n_values)

    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    game = DirectImitationGame(
        senders, receivers,
        loss, opts, baseline_type=baseline, ablation=opts.ablation,
    )

    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    epochs_to_save = list(range(opts.stats_freq, opts.n_turns * opts.n_epochs + 1, opts.stats_freq)) #+ \
                     # list(range(1, opts.n_turns * opts.n_epochs + 1, opts.stats_freq))
    epochs_to_save = sorted(epochs_to_save)

    random_state = np.random.RandomState(seed=1)
    validation_permuted = random_state.permutation(len(validation.examples))
    metrics_evaluator = CompoEvaluator(
        [validation.examples[i] for i in validation_permuted[:100]],
        opts.device,
        opts.n_attributes,
        opts.n_values,
        opts.vocab_size + 1,
        opts.distributed_context.is_distributed,
        is_population=True,
        epochs=epochs_to_save
    )

    loaders = []
    loaders.append(
        (
            "generalization hold out",
            generalization_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values, generalization=True),
        )
    )
    loaders.append(
        (
            "uniform holdout",
            uniform_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values),
        )
    )

    prefix = 'randomseed_{}'.format(opts.random_seed)
    holdout_evaluator = HoldoutEvaluator(loaders,
                                         opts.device,
                                         opts.distributed_context.is_distributed,
                                         is_population=True, epochs=epochs_to_save)
    checkpoint_saver = core.CheckpointSaver(
        checkpoint_path=opts.checkpoint_dir, checkpoint_freq=opts.checkpoint_freq, prefix=prefix)

    interactions_saver = core.InteractionSaver(
        train_epochs=epochs_to_save,
        test_epochs=epochs_to_save,
        checkpoint_dir=opts.checkpoint_dir + '/' + prefix
    )

    callbacks = [
        # core.ConsoleLogger(as_json=True, print_train_loss=True),
        metrics_evaluator,
        holdout_evaluator,
        interactions_saver,
        checkpoint_saver
    ]

    if opts.tensorboard:
        #callbacks.append(tensorboard_logger)
        pass

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        convergence_epsilon=opts.convergence_epsilon,
        callbacks=callbacks
    )

    if train_mode:
        # Initialize callbacks for first log.
        for callback in trainer.callbacks:
            callback.on_train_begin(trainer)
        # holdout_evaluator.evaluate()

        trainer.train(opts.n_turns * opts.n_epochs)

    print("---End--")

    core.close()


if __name__ == "__main__":
    import sys
    import multiprocessing as mp


    def get_args_for_string(argstring):
        master_args = sys.argv.copy()[1:]
        argstrings = list(filter(lambda elt: '--' in elt, master_args))
        args = []

        next_idx = master_args.index(argstring) + 1

        while next_idx < len(master_args) and (master_args[next_idx] not in argstrings):
            args.append(master_args[next_idx])
            next_idx += 1

        return args


    def launch_training(i, grid_dict):
        master_args = sys.argv.copy()[1:]
        args = []
        argstrings = list(filter(lambda elt: '--' in elt, master_args))

        for argstring in argstrings:
            args.append(argstring)
            if argstring in grid_dict:  # assume argstring is '--something'
                args.append(str(grid_dict[argstring]))
            else:
                next_idx = master_args.index(argstring) + 1

                while next_idx < len(master_args) and (master_args[next_idx] not in argstrings):
                    args.append(master_args[next_idx])
                    next_idx += 1

        print(master_args)

        checkpoint_dir_idx = args.index('--checkpoint_dir') + 1
        ablation = master_args[master_args.index('--ablation') + 1].split('_')[0]
        optimization = 'reinforce' if master_args[master_args.index('--imitation_reinforce')+1]=='True' else 'backprop'
        pair_sampling = 'fixed' if grid_dict['sample_pairs']=='False' else 'random'

        args[checkpoint_dir_idx] = args[checkpoint_dir_idx] + '{}_pairs/'.format(pair_sampling) + \
                                   '{}_{}/'.format(ablation, optimization) + \
                                   'pop_size_{}/n_val_{}_n_att_{}_vocab_{}_max_len_{}_arch_{}_hidden_{}_n_epochs_{}_im_weight_{}/'.format(
            grid_dict['--population_size'],
            grid_dict['--n_values'], grid_dict['--n_attributes'], grid_dict['--vocab_size'], grid_dict['--max_len'],
            grid_dict['--sender_cell'], grid_dict['--hidden'], grid_dict['--n_epochs'], grid_dict['--imitation_weight']
        )
        use_tensorboard = args[args.index('--tensorboard') + 1] == 'True'
        if use_tensorboard:
            tensorboard_dir_idx = args.index('--tensorboard_dir') + 1
            args[tensorboard_dir_idx] = args[tensorboard_dir_idx] + '/rs_{}'.format(i)

        # args.append('--random_seed')
        # args.append(str(i))
        print('FINAL ARGS')
        print(args)
        return main(args)

    # args = sys.argv.copy()[1:]
    # args.append('--random_seed')
    # args.append(0)
    # main(args)
    
    procs = [launch_training(
        i, {
            '--n_values': n_val, '--n_attributes': n_att, '--vocab_size': vocab_size, '--max_len': max_len,
            '--sender_cell': sender_cell, '--hidden': hidden, '--n_epochs': n_epochs, '--imitation_weight':im_weight,
            '--population_size': pop_size, '--imitation_reinforce': imitation_reinforce, 'sample_pairs': sample_pairs
        })
        for i in range(1)
        for n_val in get_args_for_string('--n_values')
        for n_att in get_args_for_string('--n_attributes')
        for sender_cell in get_args_for_string('--sender_cell')
        for vocab_size in get_args_for_string('--vocab_size')
        for max_len in get_args_for_string('--max_len')
        for hidden in get_args_for_string('--hidden')
        for n_epochs in get_args_for_string('--n_epochs')
        for im_weight in get_args_for_string('--imitation_weight')
        for pop_size in get_args_for_string('--population_size')
        for imitation_reinforce in get_args_for_string('--imitation_reinforce')
        for sample_pairs in get_args_for_string('--sample_pairs')
    ]


