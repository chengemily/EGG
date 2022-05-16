# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents
from egg.zoo.addition_game.architectures import Receiver, Sender
from egg.zoo.addition_game.data_readers import get_dataloaders
from egg.zoo.addition_game.callbacks import HoldoutEvaluator


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/main/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    parser.add_argument(
        "--input_size",
        type=int,
        default=20,
        help="N, where the range of inputs are 0...N-1",
    )
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        '--training_density',
        type=float,
        default=0.5,
        help='proportion of input pairs to keep in training data.'
    )
    parser.add_argument(
        '--scaling_factor',
        type=float,
        default=3,
        help='Factor by which to scale training data.'
    )
    # arguments concerning the training method
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)
    print(opts, flush=True)

    def loss(
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

    # Datasets
    train_loader, test_loader, holdout_loaders = get_dataloaders(opts.input_size, opts.training_density, opts.batch_size)

    receiver = Receiver(n_output=2 * opts.input_size - 1, n_hidden=opts.receiver_hidden)
    sender = Sender(n_hidden=opts.sender_hidden, n_inputs=2 * opts.input_size)

    sender = core.RnnSenderReinforce(
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell,
    )
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
    )

    interaction_saver = core.InteractionSaver(
            train_epochs=[1] + list(range(opts.validation_freq, opts.n_epochs, opts.validation_freq)),
            test_epochs=[1] + list(range(opts.validation_freq, opts.n_epochs, opts.validation_freq)),
            checkpoint_dir='./additions_interactions/N_{}_gen_density_{}/rs_{}/'.format(
                opts.input_size, opts.training_density, opts.random_seed)
        )
    callbacks = [
        interaction_saver,
        core.ConsoleLogger(print_train_loss=True, as_json=True),
        core.CheckpointSaver(
            checkpoint_path='./additions_checkpoints/N_{}_gen_density_{}/rs_{}/'.format(opts.input_size,
                                                                                        opts.training_density,
                                                                                                    opts.random_seed),
            checkpoint_freq=0),
    ]

    if holdout_loaders is not None:
        callbacks.append(
            HoldoutEvaluator(
                holdout_loaders,
                opts.device,
                interaction_saver
            )
        )

    optimizer = core.build_optimizer(game.parameters())
    if opts.print_validation_events == True:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
        )

    # and finally we train!
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    for training_density in [0.5]:
        for N in [5]:
            for rs in range(10):
                main(sys.argv[1:] + ['--random_seed={}'.format(rs),
                                     '--input_size={}'.format(N),
                                     '--training_density={}'.format(training_density)])
