import matplotlib.pyplot as plt
import argparse
import torch
from pathlib import Path
import typing
from typing import TypeVar, Generic, Sequence, Dict, Tuple
import pandas as pd
import numpy as np
import os
from scipy.spatial import distance

from egg.core.language_analysis import editdistance
from egg.zoo.imitation_learning.loader import load_all_interactions
from egg.zoo.imitation_learning.analysis.stats_util import *
from egg.zoo.imitation_learning.analysis.imitation_analysis import *

import scipy.stats as stats


def load_time_series_from_experiment(experiment_path, acc_thres=0.75, last_only=False):
    # load all interactions
    agged_data, sep_data = {}, {}

    for mode in ['train']:
        print(experiment_path)
        try:
            all_interactions = load_all_interactions(experiment_path, mode=mode, last_only=last_only)
            agged_data[mode], sep_data[mode], num_seeds = convert_to_acc_df(
                all_interactions,
                acc_thres=acc_thres
            )
        except FileNotFoundError:
            print('Files not found for mode={}.'.format(mode))
            continue
        except Exception as e:
            print(e)
            print('Files bad for experiment={}, mode={}'.format(experiment_path, mode))

    # Add epoch 1 to agged data val, sep data val
    # agged_data['validation'] = pd.concat([agged_data['train'].iloc[[0]], agged_data['validation']])
    # sep_data['validation'] = [pd.concat([sep_data['train'][min(i, len(sep_data['train']) - 1)].iloc[[0]], data_val])
    #                           for i, data_val in enumerate(sep_data['validation'])]

    return agged_data, sep_data, num_seeds


def reformat_tensor(tensor, population_size, batch_size=1024):
    """
    Reformats interaction to be k x N x ... (where N is the number of data points, k is the # of agents).
    Currently it is [N x k] x ...
    :param interaction:
    :param batch_size:
    :return:
    """
    cutting_points = list(np.arange(batch_size * population_size, tensor.size(0), batch_size * population_size))
    # print(cutting_points)
    tensor_split_into_batches = torch.tensor_split(tensor, cutting_points, dim=0)
    batch_sizes = [tensor.size(0) // population_size for tensor in tensor_split_into_batches]

    tensor_split_into_batches = [tensor.view(
                                        population_size,
                                        batch_sizes[i],
                                        torch.numel(tensor) // (population_size * batch_sizes[i])
                                    )
                                 for i, tensor in enumerate(tensor_split_into_batches)
                                ]
    tensor = torch.cat(tensor_split_into_batches, dim=1)
    return tensor


def get_average_levenstein_distance_for_population(messages):
    """
    :param messages: torch.Tensor of dimensions k x N x message length, where N is the dataset length. This is the
    output of all speakers at one epoch.
    :return: average levenstein distance between pairs of speakers.
    """
    pop_size = messages.shape[0]
    pair_distances = []

    for i in range(pop_size - 1):
        for j in range(i + 1, pop_size):
            pair_distances.append(get_levenstein_distance(messages[i,:100], messages[j,:100]))

    return np.mean(pair_distances)


def get_levenstein_distance(x, y):
    # x and y are two tensors of messages. Computes levenstein distance i for (x_i, y_i) and returns the average
    # over i. i indexes an input.
    levenstein_dist = lambda x, y: editdistance(x, y) / ((len(x) + len(y)) / 2)
    pairwise_distances = list(map(levenstein_dist, x, y))
    return np.mean(pairwise_distances)


def reformat_interaction(interaction):
    population_size = int(max(interaction.aux['pairings']) + 1) # maximum agent ID + 1
    # print('pop size: ', population_size)

    # print('sender input', interaction.sender_input.shape)
    interaction.sender_input = reformat_tensor(interaction.sender_input, population_size)

    # print('message', interaction.message.shape)
    interaction.message = reformat_tensor(interaction.message, population_size)

    # print('receiver output: ', interaction.receiver_output.shape)
    interaction.receiver_output = reformat_tensor(interaction.receiver_output, population_size)

    # print('receiver sample: ', interaction.receiver_sample.shape)
    interaction.receiver_sample = reformat_tensor(interaction.receiver_sample, population_size)

    for field in ['acc', 'acc_or', 'loss', 'sender_entropy', 'receiver_entropy']:
        interaction.aux[field] = reformat_tensor(interaction.aux[field], population_size)


def nested_dict(nested_dict):
    reformed_dict = {}
    for outerKey, innerDict in nested_dict.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey,
                           innerKey)] = values

    return reformed_dict


def convert_to_acc_df(all_interactions, acc_thres=-1):
    """
    :param all_interactions:
    :param acc_thres:
    :return:
    """
    all_interactions, epochs = all_interactions
    results = []
    global_results_all = []

    # convert to df
    for rs in all_interactions:
        pop_size = int(max(rs[0].aux['pairings']) + 1)

        result_dict_all = {i: {} for i in range(pop_size)}
        global_results = {'message distance': []}

        for i, epoch in enumerate(rs):
            reformat_interaction(epoch)

            # Collect messages
            global_results['message distance'].append(epoch.message)

            for metric in epoch.aux:
                if metric not in result_dict_all[0]:
                    for agent in range(pop_size):
                        result_dict_all[agent][metric] = [None] * len(epochs)
                if 'population' in metric:
                    # Collect population level metrics to global_results (population stats)
                    if metric not in global_results:
                        global_results[metric] = []
                    global_results[metric].append(epoch.aux[metric][0].item())
                for agent in range(pop_size):
                    if type(epoch.aux[metric].size()) == int or len(epoch.aux[metric].size()) == 1:
                        if 'population' not in metric:
                            # Collect agent level metrics to result_dict_all (agent level stats)
                            result = epoch.aux[metric][agent]
                            result_dict_all[agent][metric][i] = result.item()
                    else:
                        result = epoch.aux[metric][agent, :]

                        if len(result) > 1:
                            if metric == 'sender_entropy':
                                result_dict_all[agent][metric][i] = float(torch.mean(result[:-1]))
                            elif metric in ['acc', 'acc_or', 'loss'] or 'imitation' in metric:
                                result_dict_all[agent][metric][i] = float(torch.mean(result))
                        else:
                            result_dict_all[agent][metric][i] = result.item()
                    result_dict_all[agent] = {k: v for k, v in result_dict_all[agent].items() if len(v)}

        results.append(result_dict_all)

        # Compute avg levenstein distance for population for each epoch
        global_results['message distance'] = list(map(get_average_levenstein_distance_for_population,
                                                  global_results['message distance']))
        for metric in ['acc', 'acc_or', 'compo_metrics/topographic_sim', 'compo_metrics/bag_of_symbol_disent', 'sender_entropy']:
            metric_name = metric.split('/')[1] if '/' in metric else metric
            global_results['population_{}'.format(metric_name)] = np.mean(
                [result_dict_all[agent][metric] for agent in result_dict_all],
                axis=0)

        global_results_all.append(global_results)


    global_results_all = pd.concat([pd.DataFrame(global_result) for global_result in global_results_all])
    global_results = global_results_all.groupby(global_results_all.index).agg(['mean', 'std', 'max'])
    global_results['epoch'] = epochs

    separate_data = [pd.DataFrame(nested_dict(result)) for result in results \
                     if np.min([np.max(agent['acc']) for _, agent in result.items()]) > acc_thres]
    all_data = pd.concat(separate_data)
    all_data = all_data.groupby(all_data.index).agg(['mean', 'std', 'max'])
    all_data = all_data.head(len(epochs))
    all_data['epoch'] = epochs
    all_data = all_data.set_index('epoch').join(global_results.set_index('epoch'))
    all_data = all_data.reset_index()
    # print(all_data.columns)
    # print(all_data.head(10))
    # input()
    return all_data, separate_data, len(separate_data)


def plot_means_per_agent(ylabel, savepath, ploty, plotx, agged_data_dict, rs_counts, pop_size, xlabel='Epoch', mode='train'):
    # agged data list = dict{expertiment_name: {hyperparams_setting_name: {train: df, validation: df}}}
    print('plotting {}'.format(ylabel))

    ending_epochs = []
    for experiment in agged_data_dict:
        experiment_dict = agged_data_dict[experiment]
        for hyperparam_name in experiment_dict:
            hyperparam_dict = experiment_dict[hyperparam_name]
            df_for_population = hyperparam_dict[mode]
            num_rs = rs_counts[experiment][hyperparam_name]

            for agent_number in range(pop_size):
                label = '{}; agent {}; n={}'.format(experiment, agent_number,
                                                    hyperparam_name.split('im_weight')[1].split('_')[1],
                                                    num_rs
                                                    )
                df_to_plot = df_for_population[[col for col in df_for_population.columns if str(agent_number) in str(col)]]
                df_to_plot.columns = pd.MultiIndex.from_tuples(df_to_plot.columns, names=['agent', 'metric', 'stat'])
                df_to_plot = df_to_plot[agent_number]
                df_to_plot['epoch'] = df_for_population['epoch']
                plot(ylabel, ploty, plotx, agged_data=df_to_plot, label=label, xlabel=xlabel,
                     error_range=1, agg_mean_style='--')

            ending_epochs.append(max(df_for_population['epoch']))

    plt.xlim(right=min(ending_epochs))
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_population_stats(ylabel, savepath, ploty, plotx, agged_data_dict, rs_counts, pop_size, xlabel='Epoch', mode='train'):
    # agged data list = dict{expertiment_name: {hyperparams_setting_name: {train: df, validation: df}}}
    print('plotting {}'.format(ylabel))

    ending_epochs = []
    for experiment in agged_data_dict:
        experiment_dict = agged_data_dict[experiment]
        for hyperparam_name in experiment_dict:
            hyperparam_dict = experiment_dict[hyperparam_name]
            df_for_population = hyperparam_dict[mode]
            num_rs = rs_counts[experiment][hyperparam_name]
            label = '{}; {}={}; # agents={}; n={}'.format(experiment, r'$ \alpha $',
                                                     float(hyperparam_name.split('im_weight')[1].split('_')[1][:-1]),
                                                     pop_size, num_rs
                                                )
            plot(ylabel, ploty, plotx, agged_data=df_for_population, label=label, xlabel=xlabel, error_range=1, agg_mean_style='--')
            ending_epochs.append(max(df_for_population['epoch']))

    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(right=min(ending_epochs))
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_composite(agged_datas, sep_datas, n, args) -> None:
    # load all interactions
    print(args.experiment)
    experiment_name = '_'.join([experiment.replace(' ', '_').replace('/', '_') for experiment in args.experiment])
    # agged_data_trains = [agged_datas[experiment] for experiment in args.experiment] # list of dicts.
    # agged_data_trains = [run['train'] for run_list in agged_data_trains for run in run_list]

    for metric in ['context_independence', 'positional_disent', 'bag_of_symbol_disent', 'topographic_sim']:
        plot_means_per_agent(metric, 'images/direct_imitation_{}/training_{}_n_epochs_{}_composite_{}.png'.format(
            args.pairs_sampling,
            metric, args.n_epochs[0], '_'.join(experiment_name)
        ),
                   'compo_metrics/{}'.format(metric), 'epoch', agged_datas, n, 2)

    for metric in ['generalization hold out/', 'uniform holdout/', '']:
        name = metric.split(' ')[0] if len(metric) else metric

        for accuracy in ['acc', 'acc_or']:
            plot_means_per_agent(metric, 'images/direct_imitation_{}/{}_{}_n_epochs_{}_composite_{}.png'.format(
                args.pairs_sampling,
                name, accuracy,args.n_epochs[0], experiment_name),
                   '{}{}'.format(metric,accuracy), 'epoch', agged_datas, n, 2)

    plot_means_per_agent('sender entropy', 'images/direct_imitation_{}/training_n_epochs_{}_sender_entropy_composite_{}.png'.format(
        args.pairs_sampling,
        args.n_epochs[0], experiment_name
    ),
               'sender_entropy', 'epoch', agged_datas, n, 2)

    for pop_metric in ['message distance', 'generalization hold out/population_acc', 'generalization hold out/population_acc_or',
                       'uniform holdout/population_acc', 'uniform holdout/population_acc_or', 'population_acc', 'population_acc_or',
                       'population_topographic_sim', 'population_bag_of_symbol_disent', 'population_sender_entropy']:
        metric_name = pop_metric.replace(' ', '_').replace('/', '_')
        plot_population_stats(pop_metric, 'images/direct_imitation_{}/training_{}_n_epochs_{}_composite_{}.png'.format(
            args.pairs_sampling,
            metric_name, args.n_epochs[0], experiment_name
        ), pop_metric, 'epoch', agged_datas, n, 2)

def load_all_time_series(args, last_only=False) -> Tuple[Dict[str, list]]:
    """
    :return:
    agged_datas = {"control" : [df_setting_1, df_setting_2, ..]}
    sep_datas = pareil
    n = {"control": {"setting_1": n}}
    """
    agged_datas = {}
    sep_datas = {}
    n = {}

    for experiment, run_paths in get_interaction_paths(args).items():
        agged_datas[experiment] = {}
        sep_datas[experiment] = {}
        n[experiment] = {}

        for run_path in run_paths:
            agged_data, sep_data, num_seeds = load_time_series_from_experiment(
                run_path, acc_thres=args.acc_thrs, last_only=last_only
            )
            if experiment == 'kl':
                print(sep_data)
            n[experiment][run_path] = num_seeds
            agged_datas[experiment][run_path] = agged_data
            sep_datas[experiment][run_path] = sep_data

    return agged_datas, sep_datas, n


def composite_plot_metric_wrt_alpha(agged_datas, sep_datas, n, args):
    # y-axis = metric
    # x-axis = log scale of alpha.

    print(agged_datas)
    experiment_name = '_'.join([experiment.replace(' ', '_').replace('/', '_') for experiment in args.experiment])

    for metric in ['population_bag_of_symbol_disent', 'population_topographic_sim']:
        plot_metric_wrt_alpha(metric, 'images/direct_imitation_{}/alpha_{}_n_epochs_{}_composite_{}.png'.format(
            args.pairs_sampling,
            metric, args.n_epochs[0], experiment_name
        ), metric, agged_datas, n)


    plot_metric_wrt_alpha(
        'population_sender_entropy',
        'images/direct_imitation_{}/alpha_n_epochs_{}_sender_entropy_composite_{}.png'.format(
                args.pairs_sampling,
                args.n_epochs[0], experiment_name
            ),
        'population_sender_entropy', agged_datas, n)

    for pop_metric in ['message distance', 'generalization hold out/population_acc',
                       'generalization hold out/population_acc_or',
                       'uniform holdout/population_acc', 'uniform holdout/population_acc_or', 'population_acc',
                       'population_acc_or']:
        metric_name = pop_metric.replace(' ', '_').replace('/', '_')
        plot_metric_wrt_alpha(pop_metric,
                              'images/direct_imitation_{}/alpha_{}_n_epochs_{}_composite_{}.png'.format(
                                  args.pairs_sampling,
                                  metric_name, args.n_epochs[0], experiment_name
                                  ), pop_metric, agged_datas, n)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from imitation experiment.')
    parser.add_argument('--pairs_sampling',
                        type=str, default='fixed', choices=['fixed', 'random'])
    parser.add_argument('--runs_path',
                        type=str, default='/ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/direct_imitation/')
    parser.add_argument('--filter',
                        type=bool, default=True)
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['control/pop_size_2', 'receiver_backprop/pop_size_2']
                        )
    parser.add_argument('--n_val',
                        nargs='*',
                        default=[10])
    parser.add_argument('--n_att',
                        nargs='*',
                        default=[6])
    parser.add_argument('--vocab',
                        nargs='*',
                        default=[10])
    parser.add_argument('--max_len',
                        nargs='*',
                        default=[10])
    parser.add_argument('--hidden',
                        nargs='*',
                        default=[128]
                        )
    parser.add_argument('--n_epochs',
                        nargs='*',
                        default=[50])
    parser.add_argument('--acc_thrs',
                        type=float,
                        default=-0.01)
    parser.add_argument('--imitation_weight',
                        nargs='*',
                        # default=['1.0'],
                        # default=['0.0001', '0.001']
                        default=["0.000001", "0.00001", "0.0001", "0.001", "0.01"]#, "0.1"]#, "1.0"]#"0.025", "0.05", "0.1", "1.0"]
                        ),
    parser.add_argument('--alpha_plot',
                        type=bool,
                        default=True
                        )
    args = parser.parse_args()
    args.runs_path += '{}_pairs/'.format(args.pairs_sampling)

    agged_datas, sep_datas, n = load_all_time_series(args, last_only=args.alpha_plot)
    if not args.alpha_plot:
        plot_composite(agged_datas, sep_datas, n, args)
    else:
    # print(agged_datas)
        composite_plot_metric_wrt_alpha(agged_datas, sep_datas, n, args)