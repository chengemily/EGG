import copy
import argparse
import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.core import CheckpointSaver
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
from egg.zoo.compo_vs_generalization import train as compo_vs_generalization
from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.util import *

def eval_expert(metadata_path: str):
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    compo_vs_generalization.main(params, train_mode=False)


def eval_bc_prediction(new_sender, new_receiver, trainer, t=None, checkpoint_path=None):
    _, interaction = trainer.eval()
    r_loss, r_acc = new_receiver.score(interaction, val=True)
    s_loss, s_acc = new_sender.score(interaction, val=True)
    print('Epoch: {}; Receiver val loss: {}; Sender val loss: {}'.format(t, r_loss, s_loss))

    return r_loss, s_loss, r_acc, s_acc


def eval_expert_original_task(trainer):
    # print('About to evaluate og agents on og task')
    mean_loss, interaction = trainer.eval()
    acc_or, acc = interaction.aux['acc_or'].mean(), interaction.aux['acc'].mean()
    print('Expert Loss: {}. Acc_or: {}. Acc: {}'.format(mean_loss, acc_or, acc))
    # input()
    return mean_loss, acc, acc_or


def eval_bc_original_task(new_trainer, t=None, checkpoint_path=None):
    mean_loss, interaction = new_trainer.eval()
    acc_or, acc = interaction.aux['acc_or'].mean(), interaction.aux['acc'].mean()

    print('Epoch: {}; Original Task Loss: {}. Acc_or: {}. Acc: {}'.format(t, mean_loss, acc_or, acc))
    return mean_loss, acc, acc_or # new results


def train_bc(bc_args, new_sender, new_receiver, optimizer_s, optimizer_r, trainer,
             new_trainer=None, imitation=False, perf_log=None):
    new_receiver_converged = False
    new_sender_converged = False
    receiver_converged_epoch, sender_converged_epoch = 0, 0
    cumu_r_loss, cumu_s_loss = 0, 0
    cumu_r_acc, cumu_s_acc = [], []

    for t in range(bc_args.n_epochs_bc):
        val = t % bc_args.val_interval == 0
        if val:
            new_sender.eval()
            new_receiver.eval()
            r_loss, s_loss, r_acc, s_acc = eval_bc_prediction(new_sender, new_receiver, trainer, t)
            if new_trainer is not None: mean_loss, acc, acc_or = eval_bc_original_task(new_trainer, t)

            if perf_log is not None:
                log_performance(perf_log, r_loss.item(),
                            s_loss.item(), r_acc.item(), s_acc.item(), mean_loss,
                            acc.item(), acc_or.item(), sender_converged_epoch, receiver_converged_epoch)

        if imitation:
            trainer.game.train()

        _, interaction = trainer.eval(trainer.train_data, imitation=imitation)

        if not new_receiver_converged:
            new_receiver.train()
            r_loss, r_acc = train_epoch(optimizer_r, new_receiver, interaction, expert=trainer.game.receiver if imitation else None)
            cumu_r_loss += r_loss
            cumu_r_acc.append(r_acc)
            new_receiver_converged = get_grad_norm(new_receiver) < bc_args.convergence_epsilon
            receiver_converged_epoch = t
        if not new_sender_converged:
            new_sender.train()
            s_loss, s_acc = train_epoch(optimizer_s, new_sender, interaction, expert=trainer.game.sender if imitation else None)
            cumu_s_loss += s_loss
            cumu_s_acc.append(s_acc)
            new_sender_converged = get_grad_norm(new_sender) < bc_args.convergence_epsilon
            sender_converged_epoch = t

        if new_receiver_converged and new_sender_converged:
            print('Both receiver and sender gradients < epsilon={}'.format(bc_args.convergence_epsilon))
            break
        print('Epoch: {}; Receiver loss: {}; Sender loss: {}; R acc: {}; S acc: {}'.format(t, r_loss, s_loss, r_acc, s_acc))

    return cumu_s_loss, cumu_r_loss, t, s_acc, r_acc, cumu_s_acc, cumu_r_acc



def main(metadata_path: str, bc_params, expert_seed):
    bc_args = get_bc_params(bc_params)
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)

    params = checkpoint_wrapper['params']
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    params = list(filter(lambda x: 'random_seed' not in x, params))
    params.append('--random_seed={}'.format(bc_args.bc_random_seed))
    opts = get_params(params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # New agents
    new_sender, new_receiver = bc_agents_setup(opts, device, *define_agents(opts))
    optimizer_r = torch.optim.Adam(new_receiver.parameters(), lr=opts.lr)
    optimizer_s = torch.optim.Adam(new_sender.parameters(), lr=opts.lr)

    # Dataloader
    trainer = expert_setup(opts)
    new_trainer = copy.deepcopy(trainer)
    new_trainer.game.sender, new_trainer.game.receiver = new_sender.agent, new_receiver.agent

    # Logging
    perf_log = {
        'r_loss': [],
        's_loss': [],
        'r_acc': [],
        's_acc': [],
        'mean_loss': [],
        'acc': [],
        'acc_or': [],
        'epoch': [],
        'epoch_speaker': [],
        'epoch_receiver': []
    }

    train_bc(bc_args, new_sender, new_receiver, optimizer_s, optimizer_r, trainer, new_trainer, perf_log)

    # Last validation score
    print('==============================================')
    print('Last validation score')
    r_loss, s_loss, r_acc, s_acc = eval_bc_prediction(new_sender, new_receiver, trainer, t=t)

    # Integrate with og environment on validation
    print('Last validation score on original task')
    mean_loss, acc, acc_or = eval_bc_original_task(new_trainer, t=t)

    # Original model score
    print('Expert validation on original task')
    eval_expert_original_task(trainer)

    log_performance(perf_log, r_loss.item(), s_loss.item(), r_acc.item(), s_acc.item(), mean_loss, acc.item(), acc_or.item(), sender_converged_epoch,
                    receiver_converged_epoch)

    # Save BC model
    if bc_args.save_bc:
        save_behavioral_clones(bc_args, params, new_receiver, new_sender,
                               optimizer_r, optimizer_s, metadata_path, perf_log, expert_seed)

    core.close()


def train_epoch(optimizer, agent, interaction, expert=None):
    optimizer.zero_grad()
    loss, acc = agent.score(interaction, expert=expert)
    loss.backward()
    optimizer.step()
    return loss, acc


if __name__=='__main__':
    import sys
    import random

    # for i in range(100):
    #     try:
    #         resave_compo_metrics_on_whole_dataset('saved_models/' +
    #                                               'n_val_10_n_att_2_vocab_100_max_len_3_hidden_500/' +
    #                                               'checkpoint_wrapper_randomseed{}.pkl'.format(i))
    #     except:
    #         continue

    # # run program for all the things
    for seed in range(101, 131):
        print('Random seed: ', seed)
        random.seed(seed)
        params = sys.argv[1:].copy()
        params.append('--bc_random_seed={}'.format(seed))
        for i in range(100):
            try:
                main('saved_models/' +
                     'n_val_10_n_att_2_vocab_100_max_len_3_hidden_500/' +
                     'checkpoint_wrapper_randomseed{}.pkl'.format(i), params, i)
            except(FileNotFoundError):
                continue

