import copy
import argparse
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
from egg.zoo.compo_vs_generalization.train import *
from experiments.loader import *


def eval(metadata_path: str):
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    main(params, train_mode=False)

def get_bc_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs_bc", type=int, default=10, help="Number of epochs for BC training")
    parser.add_argument(
        "--early_stopping_thr_bc",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--save_bc",
        type=bool,
        default=False,
        help="Set True if you want model to be saved",
    )

    args = core.init(arg_parser=parser, params=params)
    return args

def expert_setup(opts):
    generalization_holdout_loader, uniform_holdout_loader, full_data_loader, train_loader, validation_loader, \
        train, validation = load_data(opts)
    old_sender, old_receiver = define_agents(opts)

    loss = DiffLoss(opts.n_attributes, opts.n_values)

    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    game = core.SenderReceiverRnnReinforce(
        old_sender,
        old_receiver,
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

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            metrics_evaluator,
            holdout_evaluator
        ],
    )
    return trainer


def main(metadata_path: str, bc_params):
    bc_args = get_bc_params(bc_params)
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    opts = get_params(params)
    print('Checkpoint wrapper:', checkpoint_wrapper['checkpoint_path'])

    # New agents
    new_sender, new_receiver = define_agents(opts)
    optimizer_r = torch.optim.Adam(new_receiver.parameters(), lr=opts.lr)
    optimizer_s = torch.optim.Adam(new_sender.parameters(), lr=opts.lr)

    # Dataloader
    trainer = expert_setup(opts)

    for t in range(bc_args.n_epochs):
        optimizer_r.zero_grad()
        optimizer_s.zero_grad()

        _, interaction = trainer.eval(trainer.train_data)

        # Receiver loss
        receiver_output, log_prob_r, entropy_r = new_receiver(interaction.message)
        batch_size = receiver_output.shape[0]
        receiver_output = receiver_output.view(
            batch_size, opts.n_values, opts.n_attributes
        )
        interaction.receiver_output = interaction.receiver_output.view(
            batch_size, opts.n_values, opts.n_attributes
        ).argmax(dim=1)

        # CE loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html on class probabilities.
        r_loss = F.cross_entropy(receiver_output, interaction.receiver_output)

        # Sender loss
        sender_output, log_prob_s, entropy_s, class_proba_s = new_sender(interaction.sender_input)
        class_proba_s = class_proba_s.reshape(
            batch_size * opts.max_len, opts.vocab_size
        )
        # remove end token
        interaction.message = interaction.message[:,:opts.max_len]
        interaction.message = interaction.message.reshape(
            batch_size * opts.max_len
        ) - 1 # reset to 0 - 99; in PlusOneWrapper they add one.
        s_loss = F.cross_entropy(class_proba_s, interaction.message)

        print('Epoch: {}; Receiver loss: {}; Sender loss: {}'.format(t, r_loss, s_loss))

        r_loss.backward()
        s_loss.backward()
        optimizer_r.step()
        optimizer_s.step()


if __name__=='__main__':
    import sys
    main('saved_models/checkpoint_wrapper_randomseed1.pkl', sys.argv[1:])

