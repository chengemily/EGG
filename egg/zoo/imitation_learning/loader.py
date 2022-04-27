import torch
import argparse

import pickle
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

from egg.zoo.imitation_learning.bc_archs import *


def load_bc_checkpoints(from_rs=0, to_rs=101):
    with open('/home/echeng/EGG/bc_checkpoints/bc_randomseed_{}_from_randomseed_{}_metadata.pkl'.format(to_rs, from_rs), 'rb') as f:
        x = pickle.load(f)

    # y = torch.load(x['checkpoint'])
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    return x

def log_performance(perf_log, r_loss, s_loss, r_acc, s_acc, mean_loss, acc, acc_or, t_speaker, t_receiver):
    perf_log['r_loss'].append(r_loss)
    perf_log['s_loss'].append(s_loss)
    perf_log['r_acc'].append(r_acc)
    perf_log['s_acc'].append(s_acc)
    perf_log['mean_loss'].append(mean_loss)
    perf_log['acc'].append(acc)
    perf_log['acc_or'].append(acc_or)
    perf_log['epoch'].append(max(t_speaker, t_receiver))
    perf_log['epoch_speaker'].append(t_speaker)
    perf_log['epoch_receiver'].append(t_receiver)

def save_behavioral_clones(bc_args, params, new_receiver, new_sender, optimizer_r, optimizer_s, metadata_path, metrics, expert_seed):
    file_prefix = '/home/echeng/EGG/bc_checkpoints/bc_randomseed_{}_from_randomseed_{}'.format(
        bc_args.bc_random_seed, expert_seed)

    torch.save({
       'receiver': new_receiver.state_dict(),
       'sender': new_sender.state_dict(),
       'optimizer_r': optimizer_r.state_dict(),
       'optimizer_s': optimizer_s.state_dict()
    }, file_prefix + '_checkpoint.tar')

    # update params
    params[-1] = '--random_seed={}'.format(expert_seed)

    # save torch
    with open(file_prefix + '_metadata.pkl', 'wb') as f:
        save_file = {
            'checkpoint': file_prefix + '_checkpoint.tar',
            'bc_args': bc_args,
            'params': params,
            'metadata_path_for_original_model': metadata_path,
            'perf_log': metrics
        }
        pickle.dump(save_file, f)

    print('done saving')

def load_metadata_from_pkl(filename: str):
    """
    Loads metadata from json, including model checkpoint.
    :param filename: ending in json, metadata object
    :return: json of all objects including checkpointed game
    """
    metadata = pickle.load(open(filename, 'rb'))

    return metadata

def resave_compo_metrics_on_whole_dataset(metadata_path: str):
    """
    Loads metadata of expert from json, including model checkpoint. Re-evaluates compositionality on the entire training
    and validation sets. Saves the new compositionality metrics.
    :param filename:
    """
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    ckpt = checkpoint_wrapper['checkpoint_path'].split('/')
    ckpt = '/'.join([ckpt[0], 'n_val_10_n_att_2_vocab_100_max_len_3_hidden_500', ckpt[1]])
    params.append('--load_from_checkpoint={}'.format(ckpt))
    params.remove('--sender_hidden=500')
    params.remove('--receiver_hidden=500')
    params.append('--hidden=500')
    opts = get_params(params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    trainer = expert_setup(opts)

    metrics_evaluator = trainer.callbacks[1]
    metrics_evaluator.trainer = trainer
    metrics_evaluator.dump_stats()

    with open(metadata_path, 'wb') as f:
        checkpoint_wrapper['params'] = params
        checkpoint_wrapper['checkpoint_path'] = ckpt
        checkpoint_wrapper['last_validation_compo_metrics'] = metrics_evaluator.stats
        pickle.dump(checkpoint_wrapper, f)

def get_bc_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs_bc", type=int, default=100, help="Number of epochs for BC training")
    parser.add_argument(
        "--early_stopping_thr_bc",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--convergence_epsilon",
        type=float,
        default=1e-2, # prev: 1e-4 for bc experiments.
        help="Stop training when gradient norm is less than epsilon."
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=10,
        help="Log validation results once per x epochs",
    )
    parser.add_argument(
        "--save_bc",
        type=bool,
        default=True,
        help="Set True if you want model to be saved",
    )
    parser.add_argument(
        "--bc_random_seed",
        type=int,
        default=101
    )

    args = core.init(arg_parser=parser, params=params)
    return args


def expert_setup(opts):
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

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=False),
        metrics_evaluator,
        holdout_evaluator,
    ]

    if opts.tensorboard: callbacks.append(core.TensorboardLogger())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
    )
    return trainer

def bc_agents_setup(opts, device, new_sender, new_receiver):
    return RnnSenderBC(new_sender, opts, device), RnnReceiverBC(new_receiver, opts, device)

def load_gradients(opts):
    """
    Loads the saved fixed convergence file that matches the hyperparameters in opts.
    """
    filepath = '/home/echeng/EGG/saved_models/imitation/fixed/' + \
               'n_val_{}_n_att_{}_vocab_{}_max_len_{}_hidden_{}_n_epochs_{}/'.format(
                   opts.n_values, opts.n_attributes,
                   opts.vocab_size, opts.max_len,
                   opts.hidden, opts.n_epochs
                ) + 'checkpoint_wrapper_randomseed{}.pkl'.format(opts.random_seed)
    data = load_metadata_from_pkl(filepath)
    return dict(zip(data['imitation']['epoch'], data['imitation']['avg_sender_grad'])), \
           dict(zip(data['imitation']['epoch'], data['imitation']['avg_receiver_grad']))


if __name__=='__main__':
    load_behavioral_clones()
