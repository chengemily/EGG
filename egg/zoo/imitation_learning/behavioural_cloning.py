import copy
import argparse
import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core.util import find_lengths
from egg.core import EarlyStopperAccuracy
from egg.core import CheckpointSaver
from egg.zoo.imitation_learning.archs import (
    PlusOneWrapper,
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization import train as compo_vs_generalization
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)
from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.util import *


def eval_expert(metadata_path: str):
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    compo_vs_generalization.main(params, train_mode=False)


def eval_bc_prediction(new_sender, new_receiver, trainer, t=None, checkpoint_path=None):
    _, interaction = trainer.eval()
    r_loss, r_acc, _ = new_receiver.score(interaction, val=True)
    s_loss, s_acc, _ = new_sender.score(interaction, val=True)
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
<<<<<<< HEAD
             new_trainer=None, ablation='all', imitation=False, perf_log=None, sender_aware_weight=0.0):
    new_receiver_converged = ablation == 'sender_only'
    new_sender_converged = ablation == 'receiver_only'
=======
             new_trainer=None, imitation=False, perf_log=None, sender_aware_weight=0.0):
    new_receiver_converged = False
    new_sender_converged = False
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
    receiver_converged_epoch, sender_converged_epoch = 0, 0
    cumu_r_loss, cumu_s_loss = torch.zeros(bc_args.n_epochs_bc), torch.zeros(bc_args.n_epochs_bc)
    cumu_r_acc, cumu_s_acc = torch.empty(bc_args.n_epochs_bc), torch.empty(bc_args.n_epochs_bc)
    reinforce_loss_for_sender = torch.zeros(bc_args.n_epochs_bc)
<<<<<<< HEAD
    r_loss = s_loss = r_acc = s_acc = torch.Tensor([0.0])

    for t in range(bc_args.n_epochs_bc):
        val = t % bc_args.val_interval == 0
        if val and ablation == 'all':
            new_sender.eval()
            new_receiver.eval()
            r_loss, s_loss, r_acc, s_acc = eval_bc_prediction(new_sender, new_receiver, trainer, t)
            if new_trainer is not None:
                mean_loss, acc, acc_or = eval_bc_original_task(new_trainer, t)
=======

    for t in range(bc_args.n_epochs_bc):
        val = t % bc_args.val_interval == 0
        if val:
            new_sender.eval()
            new_receiver.eval()
            r_loss, s_loss, r_acc, s_acc = eval_bc_prediction(new_sender, new_receiver, trainer, t)
            if new_trainer is not None: mean_loss, acc, acc_or = eval_bc_original_task(new_trainer, t)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

            if perf_log is not None:
                log_performance(perf_log, r_loss.item(),
                            s_loss.item(), r_acc.item(), s_acc.item(), mean_loss,
                            acc.item(), acc_or.item(), sender_converged_epoch, receiver_converged_epoch)

        _, interaction = trainer.eval(trainer.train_data)

        trainer.game.train()

        if not new_receiver_converged:
            new_receiver.train()
            r_loss, r_acc, aux_info = train_epoch(
                optimizer_r,
                new_receiver,
                interaction,
<<<<<<< HEAD
                expert=None,#trainer.game.receiver,
=======
                expert=trainer.game.receiver,
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
                imitation=imitation,
                aux_info={'expert_sender': train.game.sender if sender_aware_weight > 0 else None,
                        'sender_aware': sender_aware_weight > 0}
            )
            reinforce_loss_for_sender[t] = aux_info['reinforce_loss']
            cumu_r_loss[t] = r_loss
            cumu_r_acc[t] = r_acc
<<<<<<< HEAD
            new_receiver_converged = r_acc >= 0.999
=======
            new_receiver_converged = get_grad_norm(new_receiver) < bc_args.convergence_epsilon
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            receiver_converged_epoch = t
        if not new_sender_converged:
            new_sender.train()
            s_loss, s_acc, _ = train_epoch(
                optimizer_s,
                new_sender,
                interaction,
<<<<<<< HEAD
                expert=None,#trainer.game.sender,
=======
                expert=trainer.game.sender,
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
                imitation=imitation
            )
            cumu_s_loss[t] = s_loss
            cumu_s_acc[t] = s_acc
<<<<<<< HEAD
            new_sender_converged = s_acc >= 0.999
=======
            new_sender_converged = get_grad_norm(new_sender) < bc_args.convergence_epsilon
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            sender_converged_epoch = t

        if new_receiver_converged and new_sender_converged:
            print('Both receiver and sender gradients < epsilon={}'.format(bc_args.convergence_epsilon))
            break
        print('Epoch: {}; Receiver loss: {}; Sender loss: {}; R acc: {}; S acc: {}'.format(t, r_loss, s_loss, r_acc, s_acc))

    cumu_s_loss += sender_aware_weight * reinforce_loss_for_sender
    cumu_s_loss = cumu_s_loss.sum()
    cumu_r_loss = cumu_r_loss.sum()
<<<<<<< HEAD
    return cumu_s_loss, cumu_r_loss, sender_converged_epoch, receiver_converged_epoch, s_acc, r_acc, cumu_s_acc, cumu_r_acc
=======
    return cumu_s_loss, cumu_r_loss, t, s_acc, r_acc, cumu_s_acc, cumu_r_acc
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0


def train_epoch(optimizer, agent, interaction, expert=None, imitation=False, aux_info={}):
    optimizer.zero_grad()
    loss, acc, aux = agent.score(interaction, expert=expert, imitation=imitation, aux_info=aux_info)
    loss.backward()
    optimizer.step()
    return loss, acc, aux


<<<<<<< HEAD
def main(bc_params):
    bc_args = get_bc_params(bc_params)

    metadata_path = '/ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/basic_correlations/saved_models/' + \
                 'checkpoint_wrapper_randomseed{}.pkl'.format(bc_args.expert_seed)

    random.seed(bc_args.bc_random_seed)
=======
def main(metadata_path: str, bc_params, expert_seed):
    bc_args = get_bc_params(bc_params)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)

    params = checkpoint_wrapper['params']
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    params = list(filter(lambda x: 'random_seed' not in x, params))
    params.append('--random_seed={}'.format(bc_args.bc_random_seed))
<<<<<<< HEAD
    opts = compo_vs_generalization.get_params(params)
=======
    opts = get_params(params)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # New agents
<<<<<<< HEAD
    new_sender, new_receiver = bc_agents_setup(opts, bc_args, device,
                                               *compo_vs_generalization.define_agents(opts))
=======
    new_sender, new_receiver = bc_agents_setup(opts, device, *define_agents(opts))
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
    optimizer_r = torch.optim.Adam(new_receiver.parameters(), lr=opts.lr)
    optimizer_s = torch.optim.Adam(new_sender.parameters(), lr=opts.lr)

    # Dataloader
    trainer = expert_setup(opts)
<<<<<<< HEAD
    new_trainer = expert_setup(opts)
=======
    new_trainer = copy.deepcopy(trainer)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
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
<<<<<<< HEAD
    cumu_s_loss, cumu_r_loss, sender_converged_epoch, receiver_converged_epoch, s_acc, r_acc, cumu_s_acc, cumu_r_acc\
            = train_bc(bc_args, new_sender, new_receiver, optimizer_s, optimizer_r, trainer, new_trainer, perf_log=perf_log)

    t = max(sender_converged_epoch, receiver_converged_epoch)
=======

    train_bc(bc_args, new_sender, new_receiver, optimizer_s, optimizer_r, trainer, new_trainer, perf_log)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

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

<<<<<<< HEAD
    log_performance(perf_log, r_loss.item(), s_loss.item(), r_acc.item(), s_acc.item(), mean_loss, acc.item(),
                    acc_or.item(), sender_converged_epoch,
=======
    log_performance(perf_log, r_loss.item(), s_loss.item(), r_acc.item(), s_acc.item(), mean_loss, acc.item(), acc_or.item(), sender_converged_epoch,
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
                    receiver_converged_epoch)

    # Save BC model
    if bc_args.save_bc:
        save_behavioral_clones(bc_args, params, new_receiver, new_sender,
<<<<<<< HEAD
                               optimizer_r, optimizer_s, metadata_path, perf_log, bc_args.expert_seed)
=======
                               optimizer_r, optimizer_s, metadata_path, perf_log, expert_seed)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

    core.close()


<<<<<<< HEAD
=======



>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
if __name__=='__main__':
    import sys
    import random

<<<<<<< HEAD
    params = sys.argv[1:]
    try:
        main(params)
    except(FileNotFoundError):
        pass
=======
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
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

