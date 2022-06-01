import matplotlib.pyplot as plt
import argparse
import torch
from pathlib import Path
import typing
from typing import TypeVar, Generic, Sequence, Dict, Tuple
import pandas as pd
import numpy as np
import os
from egg.zoo.imitation_learning.loader import load_all_interactions
from egg.zoo.imitation_learning.stats_util import *

import scipy.stats as stats


def get_interaction_paths(args) -> Dict:
    paths = {experiment: [
        args.runs_path + experiment +
        ('/n_val_{}_n_att_{}_vocab_{}_max_len_{}_hidden_{}_n_epochs_{}/'.format(
            n_val, n_att, vocab, max_len, hidden, n_epochs
        ) if experiment != 'control' else \
             '/n_val_{}_n_att_{}_vocab_{}_max_len_{}_hidden_{}/'.format(
            n_val, n_att, vocab, max_len, hidden)
         )
        for n_val in args.n_val
        for n_att in args.n_att
        for vocab in args.vocab
        for max_len in args.max_len
        for hidden in args.hidden
        for n_epochs in args.n_epochs
    ] for experiment in args.experiment}

    return paths


def convert_to_acc_df(all_interactions, acc_thres=0.75):
    """
    :param all_interactions:
    :param acc_thres:
    :return:
    """
    all_interactions, epochs = all_interactions
    results = []

    # convert to df
    for rs in all_interactions:

        result_dict = {}

        for i, epoch in enumerate(rs):
            for metric in epoch.aux:
                if metric not in result_dict:
                    result_dict[metric] = [None] * len(epochs)

                if len(epoch.aux[metric]) > 1:
                    if metric == 'sender_entropy':
                        result_dict[metric][i] = float(torch.mean(epoch.aux[metric][:-1]))
                    elif metric in ['acc', 'acc_or']:
                        result_dict[metric][i] = float(torch.mean(epoch.aux[metric]))
                else:
                    result_dict[metric][i] = epoch.aux[metric].item()

        result_dict = {k: v for k, v in result_dict.items() if len(v)}
        results.append(result_dict)

    separate_data = [pd.DataFrame(result) for result in results if np.max(result['acc']) > acc_thres]
    all_data = pd.concat(separate_data)
    all_data = all_data.groupby(all_data.index).agg(['mean', 'std', 'max'])
    all_data = all_data.head(len(epochs))
    all_data['epoch'] = epochs
    return all_data, separate_data, len(separate_data)


def plot_means(ylabel, savepath, ploty, plotx, agged_data_list, legend_labels, xlabel='Epoch'):
    print(legend_labels)

    for i, label in enumerate(legend_labels):
        plot(ylabel, ploty, plotx, agged_data=agged_data_list[i], label=label, xlabel=xlabel,
             error_range=1, agg_mean_style='--')
    plt.legend()
    plt.xlim(right=min([max(agged_data['epoch']) for agged_data in agged_data_list]))
    plt.savefig(savepath)
    plt.close()


def plot(ylabel, ploty, plotx, xlabel='Epoch', agged_data=None, sep_data=None, epochs=None, savepath=None, label='mean',
         error_range=1, agg_mean_style='k--'):

    if agged_data is not None:
        x, y, err = agged_data[plotx], agged_data[ploty, 'mean'], agged_data[ploty, 'std']
        plt.fill_between(x, y - error_range *err, y + error_range*err, alpha=0.2)
        plt.plot(x, y, agg_mean_style, label=label)

    if sep_data is not None:
        for i, run in enumerate(sep_data):
            plt.plot(epochs, run[ploty])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.ylim(bottom=0)

    if savepath is not None:
        plt.legend()
        plt.savefig(savepath)


def load_time_series_from_experiment(experiment_path, acc_thres=0.75, filter=False):
    # load all interactions
    agged_data, sep_data = {}, {}

    for mode in ['train', 'validation']:
        try:
            all_interactions = load_all_interactions(experiment_path, mode=mode)
            print('mode: ', mode)
            acc_thres = acc_thres * int(filter)
            agged_data[mode], sep_data[mode], num_seeds = convert_to_acc_df(
                all_interactions,
                acc_thres=acc_thres
            )
        except FileNotFoundError:
            print('Files not found for mode={}.'.format(mode))
            continue
        except Exception:
            print('Files bad for experiment={}, mode={}'.format(experiment_path, mode))

    # Add epoch 1 to agged data val, sep data val
    agged_data['validation'] = pd.concat([agged_data['train'].iloc[[0]], agged_data['validation']])
    print(len(sep_data['validation']))
    sep_data['validation'] = [pd.concat([sep_data['train'][min(i, len(sep_data['train']) - 1)].iloc[[0]], data_val])
                              for i, data_val in enumerate(sep_data['validation'])]

    return agged_data, sep_data, num_seeds


def load_all_time_series(args) -> Tuple[Dict[str, list]]:
    agged_datas = {}
    sep_datas = {}
    n = {}
    for experiment, run_paths in get_interaction_paths(args).items():
        agged_datas[experiment] = []
        sep_datas[experiment] = []

        for run_path in run_paths:
            agged_data, sep_data, num_seeds = load_time_series_from_experiment(
                run_path, acc_thres=args.acc_thrs, filter=args.filter
            )
            n[experiment] = num_seeds
            agged_datas[experiment].append(agged_data)
            sep_datas[experiment].append(sep_data)

    return agged_datas, sep_datas, n


def plot_composite(args: argparse.Namespace) -> None:
    # load all interactions
    agged_datas, sep_datas, n = load_all_time_series(args)
    agged_data_trains = [agged_datas[experiment] for experiment in args.experiment]
    agged_data_trains = [run['train'] for run_list in agged_data_trains for run in run_list]

    labels = [experiment + ' (n={})'.format(n[experiment]) for experiment in args.experiment]

    print(agged_data_trains[0].columns)
    for metric in ['context_independence', 'positional_disent', 'bag_of_symbol_disent', 'topographic_sim']:
        plot_means(metric, 'images/training_{}_n_epochs_{}_composite_{}.png'.format(
            metric, args.n_epochs[0], '_'.join(args.experiment)
        ),
                   'compo_metrics/{}'.format(metric), 'epoch', agged_data_trains, labels)

    for metric in ['generalization hold out/', 'uniform holdout/', '']:
    # for metric in ['']:
        name = metric.split(' ')[0] if len(metric) else metric
        plot_means(metric, 'images/{}_acc_n_epochs_{}_composite.png'.format(name, args.n_epochs[0]),
                   '{}acc'.format(metric), 'epoch', agged_data_trains, labels)
    plot_means('sender entropy', 'images/training_n_epochs_{}_sender_entropy_composite_{}.png'.format(
        args.n_epochs[0], '_'.join(args.experiment)
    ),
               'sender_entropy', 'epoch', agged_data_trains, labels)
    plot_means('imitation/sample_complexity', 'images/training_n_epochs_{}_sample_complexity_composite_{}.png'.format(
        args.n_epochs[0], '_'.join(args.experiment)
    ),
               'imitation/sample_complexity', 'epoch', agged_data_trains, labels)


def percent_runs_improved(sep_data_control, sep_data_experimental):
    assert len(sep_data_control) == len(sep_data_experimental)

    for metric in sep_data_control[0].columns:
        try:
            print('{}: % runs improved by treatment: '.format(metric), np.mean(
                [list(sep_data_experimental[i][metric])[-1] >= list(control[metric])[-1] \
                 for i, control in enumerate(sep_data_control)]))
        except Exception as e:
            print('Metric: ', metric)
            print(e)


def means_statistical_test(sep_data_control, sep_data_experimental):
    print('\nMEANS STATISTICAL TEST\n')
    for metric in sep_data_control[0].columns:
        data_control = np.array([list(control[metric])[-1] for control in sep_data_control])
        data_expe = np.array([list(experimental[metric])[-1] for experimental in sep_data_experimental])

        try:
            # proceed to nonparametric test
            w, p = stats.wilcoxon(data_expe - data_control, alternative='greater')
            print('Wilcoxon test for {} with null of equal means: '.format(metric), (w, p))
        except Exception as e:
            print('Metric: ', metric)
            print(e)


def variances_statistical_test(sep_data_control, sep_data_experimental):
    for metric in sep_data_control[0].columns:
        data_control = np.array([list(control[metric])[-1] for control in sep_data_control])
        data_expe = np.array([list(experimental[metric])[-1] for experimental in sep_data_experimental])

        try:
            is_normal = normality_test(data_control) and normality_test(data_expe)

            if is_normal:
                s, p = stats.levene(data_control, data_expe, center='mean')
            else:
                s, p = stats.levene(data_control, data_expe, center='median')

            print('Levene test for {} with null hyp of equal variances: '.format(metric), (s, p))
        except Exception as e:
            print('Metric: ', metric)
            print(e)


def data_analysis_suite(args):
    _, sep_datas = load_all_time_series(args)
    sep_data_control, sep_data_experimental = sep_datas['control'][0]['train'], sep_datas['conv'][0]['train']
    percent_runs_improved(sep_data_control, sep_data_experimental)
    means_statistical_test(sep_data_control, sep_data_experimental)
    variances_statistical_test(sep_data_control, sep_data_experimental)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from imitation experiment.')
    parser.add_argument('--runs_path',
                        type=str, default='./checkpoints/imitation/')
    parser.add_argument('--filter',
                        type=bool, default=False)
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['conv', 'control']
                        )
    parser.add_argument('--n_val',
                        nargs='*',
                        default=[10])
    parser.add_argument('--n_att',
                        nargs='*',
                        default=[2])
    parser.add_argument('--vocab',
                        nargs='*',
                        default=[100])
    parser.add_argument('--max_len',
                        nargs='*',
                        default=[3])
    parser.add_argument('--hidden',
                        nargs='*',
                        default=[500]
                        )
    parser.add_argument('--n_epochs',
                        nargs='*',
                        default=[50])
    parser.add_argument('--acc_thrs',
                        type=float,
                        default=0.9)
    args = parser.parse_args()

    plot_composite(args)
    # data_analysis_suite(args)