# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import copy
import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core.early_stopping import *
from egg.core.pcgrad import PCGrad

from egg.zoo.compo_vs_generalization.archs import (
    Freezer,
    NonLinearReceiver,
    PlusOneWrapper,
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics
from egg.zoo.compo_vs_generalization.train import *

from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.behavioural_cloning import *
from egg.zoo.imitation_learning.util import *

def imit_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kick', type=str, choices=['none', 'imitation', 'random'], default='none')
    parser.add_argument('--turn_taking', type=str, choices=['fixed', 'convergence'], default='fixed')
    parser.add_argument('--n_turns', type=int, default=50)
    parser.add_argument("--n_attributes",
                        type=int,
                        default=4, help="")
    parser.add_argument("--n_values",
                        type=int,
                        default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=100)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument(
        "--baseline", type=str, choices=["no", "mean", "builtin"], default="mean"
    )
    parser.add_argument(
        "--density_data", type=int, default=0, help="no sampling if equal 0"
    )
    parser.add_argument('--hidden',
                        type=int,
                        default=50,
                        help='Size of hidden layer of both agents')

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
        default=0.01,
        help="Stop training when gradient norm is less than epsilon."
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Set True if you want model to be saved",
    )
    parser.add_argument(
        "--pc_grad",
        type=bool,
        default=False,
        help="Set True if you want to use PC Grad"
    )

    args = core.init(arg_parser=parser, params=params)

    return args

def log_perf(perf_log, curr_epoch, metrics_evaluator, holdout_evaluator, trainer, tensorboard_logger=None):
    perf_log['epoch'].append(curr_epoch)

    mean_loss, last_interaction = trainer.eval()

    metrics_evaluator.dump_stats()
    compo_metrics = metrics_evaluator.stats
    perf_log['compo_metrics'].append(compo_metrics)

    perf_log['uniformtest_acc'].append(holdout_evaluator.results["uniform holdout"]["acc"])
    perf_log['uniformtest_acc_or'].append(holdout_evaluator.results["uniform holdout"]["acc_or"])
    perf_log['generalization_acc'].append(holdout_evaluator.results['generalization hold out']['acc'])
    perf_log['generalization_acc_or'].append(holdout_evaluator.results['generalization hold out']['acc_or'])

    perf_log['acc'].append(last_interaction.aux["acc"].mean().item())
    perf_log['acc_or'].append(last_interaction.aux["acc_or"].mean().item())
    perf_log['mean_loss'].append(mean_loss)

    if tensorboard_logger is not None:
        for metric, value in compo_metrics.items():
            tensorboard_logger.writer.add_scalar('Metrics/{}'.format(metric), value, curr_epoch)

def log_perf_bc(curr_epoch, perf_log, s_loss, r_loss, samp_complexity, last_s_acc, last_r_acc,
                # avg_sender_grad, avg_receiver_grad,
                tensorboard_logger=None):
    if tensorboard_logger is not None:
        tensorboard_logger.writer.add_scalar('Metrics/bc_epochs', samp_complexity, curr_epoch)
        tensorboard_logger.writer.add_scalar('Metrics/bc_r_loss', r_loss,
                                             curr_epoch)
        tensorboard_logger.writer.add_scalar('Metrics/bc_s_loss', s_loss,
                                             curr_epoch)

    perf_log['imitation']['epoch'].append(curr_epoch)
    perf_log['imitation']['imitation_loss_s'].append(s_loss)
    perf_log['imitation']['imitation_loss_r'].append(r_loss)
    perf_log['imitation']['sample_complexity'].append(samp_complexity)
    perf_log['imitation']['imitation_acc_s'].append(last_s_acc)
    perf_log['imitation']['imitation_acc_r'].append(last_r_acc)
    # perf_log['imitation']['avg_sender_grad'].append(avg_sender_grad)
    # perf_log['imitation']['avg_receiver_grad'].append(avg_receiver_grad)


def main(params, train_mode=True):
    import copy

    opts = imit_params(params)
    bc_opts = get_bc_params(params)

    device = opts.device

    generalization_holdout_loader, uniform_holdout_loader, full_data_loader, train_loader, validation_loader, \
    train, validation = load_data(opts)
    sender, receiver = define_agents(opts)

    loss = DiffLoss(opts.n_attributes, opts.n_values)

    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
    )

    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    metrics_evaluator = Metrics(
        validation.examples,
        opts.device,
        opts.n_attributes,
        opts.n_values,
        opts.vocab_size + 1,
        freq=opts.stats_freq,
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

    prefix = 'randomseed_{}'.format(
        opts.random_seed)
    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=False)
    early_stopper_conv = EarlyStopperConvergence(opts.convergence_epsilon, validation=False)
    checkpoint_saver = core.CheckpointSaver(
        checkpoint_path=opts.checkpoint_dir, checkpoint_freq=opts.checkpoint_freq, prefix=prefix)
    tensorboard_logger = core.TensorboardLogger() if opts.tensorboard else None

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=False),
        early_stopper,
        metrics_evaluator,
        holdout_evaluator,
        checkpoint_saver
    ]

    if opts.turn_taking == 'convergence':
        callbacks.append(early_stopper_conv)

    if opts.tensorboard:
        callbacks.append(tensorboard_logger)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        convergence_epsilon=opts.convergence_epsilon,
        callbacks=callbacks
    )

    if train_mode:

        perf_log = {
            'checkpoint_path': opts.checkpoint_dir + prefix + '_final.tar',
            'epoch': [],
            'compo_metrics': [],
            'acc': [],
            'acc_or': [],
            'uniformtest_acc': [],
            'uniformtest_acc_or': [],
            'generalization_acc': [],
            'generalization_acc_or': [],
            'mean_loss': [],
            'imitation': {'epoch': [], 'sample_complexity': [], 'acc': [], 'imitation_loss_s': [], 'imitation_loss_r': [],
                          'imitation_acc_s': [], 'imitation_acc_r': [], 'avg_sender_grad': [], 'avg_receiver_grad': []},
            'params': params
        }

        # Take turns
        trainer.start_epoch = 0
        curr_epoch = 0

        # Initialize callbacks for first log.
        for callback in trainer.callbacks:
            callback.on_train_begin(trainer)
        holdout_evaluator.evaluate()

        for turns in range(opts.n_turns):
            log_perf(perf_log, curr_epoch, metrics_evaluator, holdout_evaluator, trainer, tensorboard_logger)

            # Task training
            trainer.train(trainer.start_epoch + opts.n_epochs)
            if opts.turn_taking == 'convergence':
                early_stop = max(early_stopper_conv.epoch, early_stopper.epoch)
            else:
                early_stop = early_stopper.epoch
            curr_epoch = min(early_stop, trainer.start_epoch + opts.n_epochs)
            trainer.start_epoch = curr_epoch

            # Post-log
            log_perf(perf_log, curr_epoch, metrics_evaluator, holdout_evaluator, trainer, tensorboard_logger)

            # Imitation
            bc_speaker, bc_receiver = bc_agents_setup(opts, device, *define_agents(opts))
            bc_optimizer_r = torch.optim.Adam(bc_receiver.parameters(), lr=opts.lr)
            bc_optimizer_s = torch.optim.Adam(bc_speaker.parameters(), lr=opts.lr)

            # Train bc agents until convergence, logging all the while.
            if opts.kick == 'imitation':
                optimizer.zero_grad()

            s_loss, r_loss, t, last_s_acc, last_r_acc, cumu_s_acc, cumu_r_acc = train_bc(
                bc_opts,
                bc_speaker, bc_receiver,
                bc_optimizer_s, bc_optimizer_r,
                trainer,
                imitation= (opts.kick == 'imitation')
            )

            if opts.kick == 'imitation':
                optimizer.step()

            log_perf_bc(curr_epoch, perf_log, s_loss.item(), r_loss.item(), t, last_s_acc.item(), last_r_acc.item(),
                        # 0, 0,
                        tensorboard_logger)

            if opts.kick == 'random':
                sender_grads, rcvr_grads = load_gradients(opts)
                with torch.no_grad():
                    for param in game.sender.parameters():
                        param.add_(
                            torch.randn(param.size(), device=device) * sender_grads[curr_epoch] * torch.sqrt(torch.pi / 2))
                    for param in game.receiver.parameters():
                        param.add_(
                            torch.randn(param.size(), device=device) * rcvr_grads[curr_epoch] * torch.sqrt(torch.pi / 2))

            log_perf(perf_log, curr_epoch + 1, metrics_evaluator, holdout_evaluator, trainer, tensorboard_logger)

            # Early stopping / convergence criterion.
            if early_stopper.should_stop():
                break
    else:
        loss, last_interaction = trainer.eval()

    print("---End--")

    if opts.save:
        folder_name = opts.checkpoint_dir.split('/')[1:]
        saved_models_path = '/home/echeng/EGG/saved_models/{}'.format('/'.join(folder_name))

        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)

        # save model checkpoint (see trainers.py)
        with open('{}/checkpoint_wrapper_randomseed{}.pkl'.format(
                saved_models_path,
                opts.random_seed),
                'wb') as f:
            pickle.dump(perf_log, f)

    core.close()


if __name__ == "__main__":
    import sys
    import multiprocessing as mp


    def get_args_for_string(argstring):
        master_args = sys.argv.copy()[1:]
        print(master_args)
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

        checkpoint_dir_idx = args.index('--checkpoint_dir') + 1
        args[checkpoint_dir_idx] = args[checkpoint_dir_idx] + 'n_val_{}_n_att_{}_vocab_{}_max_len_{}_hidden_{}_n_epochs_{}/'.format(
            grid_dict['--n_values'], grid_dict['--n_attributes'], grid_dict['--vocab_size'], grid_dict['--max_len'],
            grid_dict['--hidden'], grid_dict['--n_epochs']
        )
        use_tensorboard = args[args.index('--tensorboard') + 1] == 'True'

        if use_tensorboard:
            tensorboard_dir_idx = args.index('--tensorboard_dir') + 1
            args[tensorboard_dir_idx] = args[tensorboard_dir_idx] + 'rs_{}'.format(i)

        args.append('--random_seed')
        args.append(str(i))

        print(args)
        return main(args)

    # args = sys.argv.copy()[1:]
    # main(args)

    pool = mp.Pool(mp.cpu_count())

    results = [
        pool.apply(launch_training, args=(i, {
            '--n_values': n_val, '--n_attributes': n_att, '--vocab_size': vocab_size, '--max_len': max_len,
            '--hidden': hidden, '--n_epochs': n_epochs
        }))
        for i in range(2)
        for n_val in get_args_for_string('--n_values')
        for n_att in get_args_for_string('--n_attributes')
        for vocab_size in get_args_for_string('--vocab_size')
        for max_len in get_args_for_string('--max_len')
        for hidden in get_args_for_string('--hidden')
        for n_epochs in get_args_for_string('--n_epochs')
    ]
    #
    pool.close()


