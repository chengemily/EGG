import matplotlib.pyplot as plt
import argparse
import torch
from pathlib import Path
import typing
from typing import TypeVar, Generic, Sequence
import pandas as pd
import numpy as np

from egg.zoo.addition_game.analysis_util import *

RANDOM = 0


def convert_to_acc_df(all_interactions, acc_thres=0.75):
    all_interactions, epochs = all_interactions
    results = []

    # convert to df
    for rs in all_interactions:
        result_dict = {'acc': [], 'sender_entropy': []}
        for epoch in rs:
            result_dict['acc'].append(float(torch.mean(epoch.aux['acc'])))
            result_dict['sender_entropy'].append(float(torch.mean(epoch.aux['sender_entropy'][:-1])))
        results.append(result_dict)

    separate_data = [pd.DataFrame(result) for result in results if np.max(result['acc']) > acc_thres]
    all_data = pd.concat(separate_data)
    all_data = all_data.groupby(all_data.index).agg(['mean', 'std', 'max'])
    all_data = all_data.head(len(epochs))
    all_data['epoch'] = epochs
    return all_data, separate_data


def plot_means(ylabel, savepath, ploty, plotx, agged_data_list, legend_labels, xlabel='Epoch'):
    for i, label in enumerate(legend_labels):
        plot(ylabel, ploty, plotx, agged_data=agged_data_list[i], label=label, xlabel=xlabel,
             error_range=1, agg_mean_style='--')
    plt.legend()
    plt.xlim(right=min([max(agged_data['epoch']) for agged_data in agged_data_list]))
    plt.savefig(savepath)
    plt.close()


def plot(ylabel, ploty, plotx, xlabel='Epoch', agged_data=None, sep_data=None, epochs=None, savepath=None, label='mean',
         error_range=2, agg_mean_style='k--'):

    if agged_data is not None:
        x, y, err = agged_data[plotx], agged_data[ploty, 'mean'], agged_data[ploty, 'std']
        plt.fill_between(x, y - error_range *err, y + error_range*err, alpha=0.4)
        plt.plot(x, y, agg_mean_style, label=label)

    if sep_data is not None:
        for i, run in enumerate(sep_data):
            plt.plot(epochs, run[ploty])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(bottom=0, top=1)

    if savepath is not None:
        plt.legend()
        plt.savefig(savepath)


def load_time_series_from_experiment(experiment_path, filter=False):
    # load all interactions
    agged_data, sep_data = {}, {}

    for mode in ['train', 'validation', 'uniform', 'permutation']:
        try:
            all_interactions = load_all_interactions(experiment_path, mode=mode)
            print('mode: ', mode)

            if mode == 'permutation':
                holdout_labels = all_interactions[0][-1][-1].labels
                print('size permutation: ', len(holdout_labels))
                probs = {}
                for label in holdout_labels:
                    if int(label) not in probs:
                        probs[int(label)] = 1
                    else:
                        probs[int(label)] += 1
                for label in probs:
                    probs[label] /= len(holdout_labels)

            acc_thres = 0.75 * int(filter) * int(mode not in ['uniform', 'permutation'])
            agged_data[mode], sep_data[mode] = convert_to_acc_df(
                all_interactions,
                acc_thres=acc_thres
            )
        except FileNotFoundError:
            print('Files not found for mode={}.'.format(mode))
            continue

    # Add epoch 1 to agged data val, sep data val
    agged_data['validation'] = pd.concat([agged_data['train'].iloc[[0]], agged_data['validation']])
    sep_data['validation'] = [pd.concat([sep_data['train'][i].iloc[[0]], data_val])
                              for i, data_val in enumerate(sep_data['validation'])]


    return agged_data, sep_data


def plot_composite_accuracy(args: argparse.Namespace) -> None:
    # load all interactions
    agged_data_trains = []
    agged_data_vals = []
    labels = []

    for experiment in args.experiment:
        agged_data, sep_data = load_time_series_from_experiment(
            args.runs_path + experiment
        )
        agged_data_trains.append(agged_data['train'])
        agged_data_vals.append(agged_data['validation'])

        # labels.append('V={}'.format(experiment.split('_')[-1]))
    labels = ['N=5', 'N=10', 'N=20']
    plot_means('Training accuracy', 'images/addition_training_acc_composite_N.png', 'acc', 'epoch', agged_data_trains, labels)
    plot_means('Validation accuracy', 'images/addition_val_acc_composite_N.png', 'acc', 'epoch', agged_data_vals, labels)


def plot_composite_generalization(args: argparse.Namespace) -> None:
    # load all interactions
    for experiment in args.experiment:
        print('experiment: ', experiment)

        agged_data, sep_data = load_time_series_from_experiment(
            args.runs_path + experiment
        )
        labels = ['train', 'validation', 'uniform', 'combination', 'permutation']

        density = experiment.split('_')[-1]

        plot_means('Accuracy', 'images/addition_acc_composite_d_{}.png'.format(density),
                   'acc', 'epoch', [agged_data[mode] for mode in
                                    labels], labels)


def print_statistics(args: argparse.Namespace) -> None:
    # load all interactions
    for experiment in args.experiment:
        print('experiment: ', experiment)

        agged_data, sep_data = load_time_series_from_experiment(
            args.runs_path + experiment, args.filter
        )

        for mode in agged_data:
            print('MODE: ', mode)
            agged_data[mode] = agged_data[mode]
            print('AVG BEST ACC: ', agged_data[mode]['acc', 'mean'].max())
            maxidx = agged_data[mode]['acc', 'mean'].argmax()
            print('MAX BEST ACC: ', agged_data[mode]['acc', 'max'].max())
            print('BEST ACC STD: ', list(agged_data[mode]['acc', 'std'])[maxidx])

        for mode in sep_data:
            print('MODE: ', mode)
            maxidxes = [
                list(agged_data[mode]['epoch'])[rs.head(len(agged_data[mode]))['acc'].argmax()] for rs in sep_data[mode]]
            print('AVG IDX of BEST ACC: ', np.mean(maxidxes))
            print('STD IDX of BEST ACC: ', np.std(maxidxes))

        best_validation_idxs = [rs.head(len(agged_data['validation']))['acc'].argmax() for rs in sep_data['validation']]

        try:
            for mode in ['uniform', 'permutation']:
                print('MODE: ', mode)
                maxidxes = [
                    list(agged_data[mode]['epoch'])[rs.head(len(agged_data[mode]))['acc'].argmax()] for rs in sep_data[mode]]
                print('AVG IDX of BEST ACC: ', np.mean(maxidxes))
                print('STD IDX of BEST ACC: ', np.std(maxidxes))
                print('PERCENT OF RUNS > RANDOM: ', np.mean([
                    int(list(rs['acc'])[best_validation_idxs[i] - 1] > RANDOM) for i, rs in enumerate(sep_data[mode])
                ]))
        except:
            print('No holdout mode.')
            pass


def plot_acc_curves(args: argparse.Namespace) -> None:
    agged_data_train, sep_data_train, agged_data_val, sep_data_val = load_time_series_from_experiment(
        args.runs_path + args.experiment
    )

    plot('Training accuracy', 'images/addition_training_acc_{}.png'.format(args.experiment), 'acc', 'epoch',
         sep_data=sep_data_train, agged_data=agged_data_train, epochs=agged_data_train['epoch'])
    plt.close()
    plot('Validation accuracy', 'images/addition_val_acc_{}.png'.format(args.experiment), 'acc', 'epoch',
         sep_data=sep_data_val, agged_data=agged_data_val, epochs=agged_data_val['epoch'])
    plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from addition experiment.')
    parser.add_argument('--runs_path',
                        type=str, default='./additions_interactions/')
    parser.add_argument('--filter',
                        type=bool, default=False)
    parser.add_argument('--input_size',
                        type=int, default=20)
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['full_6000', 'full_1000', 'full_400', 'full_40']
                        )

    args = parser.parse_args()

    RANDOM = 1 / (2 * args.input_size - 1)

    plot_acc_curves(args)
    plot_composite_accuracy(args)
    plot_composite_generalization(args)
    print_statistics(args)