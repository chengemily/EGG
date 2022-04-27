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
from egg.core import EarlyStopperAccuracy
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


def get_params(params):
    parser = argparse.ArgumentParser()
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
        "--save",
        type=bool,
        default=False,
        help="Set True if you want model to be saved",
    )

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, generalization=False):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.test_generalization:
            acc, acc_or, loss = 0.0, 0.0, 0.0

            for attr in range(self.n_attributes):
                zero_index = torch.nonzero(sender_input[:, attr, 0]).squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat(
                    [masked_input[:, :attr, :], masked_input[:, attr + 1 :, :]], dim=1
                )
                no_attribute_output = torch.cat(
                    [masked_output[:, :attr, :], masked_output[:, attr + 1 :, :]], dim=1
                )

                n_attributes = self.n_attributes - 1

                # Tensor: batch_size x n_attributes (a batch of objects, each with n attributes)
                attr_acc = (
                    # average across a batch of objects
                    (
                        # whether the listener got all the attributes right in one vector (about an object)
                        (
                            no_attribute_output.argmax(dim=-1)
                            == no_attribute_input.argmax(dim=-1)
                        ).sum(dim=1)
                        == n_attributes
                    )
                    .float()
                    .mean()
                )
                acc += attr_acc

                #
                attr_acc_or = (
                    # average score on an attribute (averaged across all attributes and objects in the batch)
                    (
                        no_attribute_output.argmax(dim=-1)
                        == no_attribute_input.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                )
                acc_or += attr_acc_or
                labels = no_attribute_input.argmax(dim=-1).view(
                    masked_size * n_attributes
                )
                predictions = no_attribute_output.view(
                    masked_size * n_attributes, self.n_values
                )
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc /= self.n_attributes
            acc_or /= self.n_attributes
        else:
            acc = (
                # whether the listener got all the attributes right in one vector (about an object), batch_size x 1
                torch.sum(
                    (
                        receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
                    ).detach(),
                    dim=1,
                )
                == self.n_attributes
            ).float()

            # whether the listener got each attribute right: batch_size x n_attributes
            acc_or = (
                receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
            ).float()

            receiver_output = receiver_output.view(
                batch_size * self.n_attributes, self.n_values
            )
            labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
            loss = (
                F.cross_entropy(receiver_output, labels, reduction="none")
                .view(batch_size, self.n_attributes)
                .mean(dim=-1)
            )

        # acc_or: in [0, 1], on average over each attribute, whether the prediction is correct
        # acc: whether the
        return loss, {"acc": acc, "acc_or": acc_or}


def load_data(opts):
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    if opts.density_data > 0:
        sampled_data = select_subset_V2(
            full_data, opts.density_data, opts.n_attributes, opts.n_values
        )
        full_data = copy.deepcopy(sampled_data)

    train, generalization_holdout = split_holdout(full_data)
    train, uniform_holdout = split_train_test(train, 0.1)

    generalization_holdout, train, uniform_holdout, full_data = [
        one_hotify(x, opts.n_attributes, opts.n_values)
        for x in [generalization_holdout, train, uniform_holdout, full_data]
    ]

    train, validation = ScaledDataset(train, opts.data_scaler), ScaledDataset(train, 1)

    generalization_holdout, uniform_holdout, full_data = (
        ScaledDataset(generalization_holdout),
        ScaledDataset(uniform_holdout),
        ScaledDataset(full_data),
    )
    generalization_holdout_loader, uniform_holdout_loader, full_data_loader = [
        DataLoader(x, batch_size=opts.batch_size)
        for x in [generalization_holdout, uniform_holdout, full_data]
    ]

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=len(validation))

    return generalization_holdout_loader, uniform_holdout_loader, full_data_loader, train_loader, validation_loader, train, validation

def define_agents(opts):
    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell in ["lstm", "rnn", "gru"]:
        receiver = Receiver(n_hidden=opts.hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.hidden,
            cell=opts.receiver_cell,
        )
    else:
        raise ValueError(f"Unknown receiver cell, {opts.receiver_cell}")

    if opts.sender_cell in ["lstm", "rnn", "gru"]:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.hidden,
            max_len=opts.max_len,
            cell=opts.sender_cell,
        )
    else:
        raise ValueError(f"Unknown sender cell, {opts.sender_cell}")

    sender = PlusOneWrapper(sender)
    return sender, receiver


def main(params, train_mode=True):
    import copy

    opts = get_params(params)
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
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)
    checkpoint_saver = core.CheckpointSaver(
        checkpoint_path=opts.checkpoint_dir, checkpoint_freq=opts.checkpoint_freq, prefix=prefix)

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=False),
        early_stopper,
        metrics_evaluator,
        holdout_evaluator,
        checkpoint_saver
    ]

    if opts.tensorboard: callbacks.append(core.TensorboardLogger())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
    )

    if train_mode:
        trainer.train(n_epochs=opts.n_epochs)
        last_interaction = early_stopper.validation_stats[-1][1]
        uniformtest_acc = holdout_evaluator.results["uniform holdout"]["acc"]
        uniformtest_acc_or = holdout_evaluator.results["uniform holdout"]["acc_or"]
        generalization_acc = holdout_evaluator.results['generalization hold out']['acc']
        generalization_acc_or = holdout_evaluator.results['generalization hold out']['acc_or']
    else:
        loss, last_interaction = trainer.eval()

    validation_acc = last_interaction.aux["acc"].mean()
    validation_acc_or = last_interaction.aux["acc_or"].mean()

    print('Validation acc: ', validation_acc)
    print('Validation acc or: ', validation_acc_or)

    print("---End--")

    if train_mode and opts.save and validation_acc.item() >= opts.early_stopping_thr:
        folder_name = opts.checkpoint_dir.split('/')[1]
        saved_models_path = '/home/echeng/EGG/saved_models/{}'.format(folder_name)

        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)

        # save model checkpoint (see trainers.py)
        with open('{}/checkpoint_wrapper_randomseed{}.pkl'.format(
                saved_models_path,
                opts.random_seed),
                  'wb') as f:
            pickle.dump({
                'checkpoint_path': opts.checkpoint_dir + prefix + '_final.tar',
                'validation_acc': validation_acc.item(),
                'validation_acc_or': validation_acc_or.item(),
                'uniformtest_acc': uniformtest_acc,
                'uniformtest_acc_or': uniformtest_acc_or,
                'generalization_acc': generalization_acc,
                'generalization_acc_or': generalization_acc_or,
                'last_validation_compo_metrics': metrics_evaluator.stats,
                'params': params
            }, f)

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
            if argstring in grid_dict: # assume argstring is '--something'
                args.append(str(grid_dict[argstring]))
            else:
                next_idx = master_args.index(argstring) + 1

                while next_idx < len(master_args) and (master_args[next_idx] not in argstrings):
                    args.append(master_args[next_idx])
                    next_idx += 1

        args.append('--checkpoint_dir')
        args.append('checkpoints/n_val_{}_n_att_{}_vocab_{}_max_len_{}_hidden_{}/'.format(
            grid_dict['--n_values'], grid_dict['--n_attributes'], grid_dict['--vocab_size'], grid_dict['--max_len'],
            grid_dict['--hidden']
        ))
        args.append('--random_seed')
        args.append(str(i))

        print(args)

        return main(args)

    pool = mp.Pool(mp.cpu_count())

    results = [
        pool.apply(launch_training, args=(i, {
            '--n_values': n_val, '--n_attributes': n_att, '--vocab_size': vocab_size, '--max_len': max_len, '--hidden': hidden
        }))
        for i in range(50)
        for n_val in get_args_for_string('--n_values')
        for n_att in get_args_for_string('--n_attributes')
        for vocab_size in get_args_for_string('--vocab_size')
        for max_len in get_args_for_string('--max_len')
        for hidden in get_args_for_string('--hidden')
    ]

    pool.close()


