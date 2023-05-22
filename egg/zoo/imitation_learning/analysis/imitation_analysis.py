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
from egg.zoo.imitation_learning.analysis.stats_util import *
from egg.core.language_analysis import *
from egg.zoo.imitation_learning.callbacks import *

import scipy.stats as stats


def get_interaction_paths(args) -> Dict:
    paths = {experiment: list(set([
        args.runs_path + experiment +
        (
        '/n_val_{}_n_att_{}_vocab_{}_max_len_{}_arch_gru_hidden_{}_n_epochs_{}_im_weight_{}/'.format(
            n_val, n_att, vocab, max_len, hidden, n_epochs, im_weight
        )
         if 'control' not in experiment else \
             '/n_val_{}_n_att_{}_vocab_{}_max_len_{}_arch_gru_hidden_{}_n_epochs_{}_im_weight_0.0/'.format(
            n_val, n_att, vocab, max_len, hidden, n_epochs)
         )
        for n_val in args.n_val
        for n_att in args.n_att
        for vocab in args.vocab
        for max_len in args.max_len
        for hidden in args.hidden
        for n_epochs in args.n_epochs
        for im_weight in args.imitation_weight
    ])) for experiment in args.experiment}
    # print(paths)
    return paths


def experiment_path_to_dict(experiment_path):
    field_value_dict = {}
    for field in ['imitation_weight']:
        value = experiment_path.split(field)[1].split('_')[1]
        field_value_dict[field] = value

    print(field_value_dict)
    return field_value_dict


def convert_to_acc_df(all_interactions, acc_thres=-1):
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
                if epoch.aux[metric] is None or 'expert_message' in metric:
                    continue

                if metric not in result_dict:
                    if 'imitator_message' not in metric:
                        result_dict[metric] = [None] * len(epochs)
                    else:
                        expert_name = metric.split('_')[0]
                        for compo in ['topographic_sim', 'positional_disent', 'bag_of_symbol_disent', 'context_independence']:
                            result_dict[expert_name + '_' + compo] = [None] * len(epochs)

                # if 'imitator_message' in metric:
                #     expert_name = metric.split('_')[0]
                #     # compute compo metrics
                #     result_dict[expert_name + '_topographic_sim'][i] = TopographicSimilarity.compute_topsim(
                #         epoch.sender_input[:100], epoch.aux[metric][:100])
                #     result_dict[expert_name + '_positional_disent'][i] = Disent.posdis(epoch.sender_input[:100],
                #                                                             epoch.aux[metric][:100])
                #     result_dict[expert_name + '_bag_of_symbol_disent'][i] = Disent.bosdis(epoch.sender_input[:100],
                #                                                             epoch.aux[metric][:100], 10)
                #     result_dict[expert_name + '_context_independence'][i] = context_independence(6, 10,
                #                                                                epoch.sender_input[:100],
                #                                                                epoch.aux[metric][:100], 'cpu')

                try:
                    len(epoch.aux[metric])
                except:
                    continue
                if len(epoch.aux[metric]) > 1:
                    if metric == 'sender_entropy':
                        result_dict[metric][i] = float(torch.mean(epoch.aux[metric][:-1]))
                    elif 'acc' in metric or 'acc_or' in metric or 'imitation' in metric or 'loss' in metric:
                        result_dict[metric][i] = float(torch.mean(epoch.aux[metric]))
                else:
                    result_dict[metric][i] = epoch.aux[metric].item()

        result_dict = {k: v for k, v in result_dict.items() if len(v)}
        results.append(result_dict)

    separate_data = [pd.DataFrame(result) for result in results] #if np.max(result['acc']) > acc_thres]
    all_data = pd.concat(separate_data)
    all_data = all_data.groupby(all_data.index).agg(['mean', 'std', 'max'])
    all_data = all_data.head(len(epochs))
    all_data['epoch'] = epochs

    return all_data, separate_data, len(separate_data)


def plot_means(ylabel, savepath, ploty, plotx, agged_data_dict, rs_counts, xlabel='Epoch', mode='train'):
    # agged data list = dict{expertiment_name: {hyperparams_setting_name: {train: df, validation: df}}}
    print('plotting {}'.format(ylabel))

    ending_epochs = []
    for experiment in agged_data_dict:
        experiment_dict = agged_data_dict[experiment]
        for hyperparam_name in experiment_dict:
            hyperparam_dict = experiment_dict[hyperparam_name]
            df_to_plot = hyperparam_dict[mode]
            num_rs = rs_counts[experiment][hyperparam_name]
            label = '{}; im_weight={}; n={}'.format(experiment,
                                                    hyperparam_name.split('im_weight')[1].split('_')[1],
                                                    num_rs
                                                    )
            plot(ylabel, ploty, plotx, agged_data=df_to_plot, label=label, xlabel=xlabel,
                 error_range=1, agg_mean_style='--')

            ending_epochs.append(max(df_to_plot['epoch']))

    # for i, label in enumerate(legend_labels):
    #     plot(ylabel, ploty, plotx, agged_data=agged_data_list[i], label=label, xlabel=xlabel,
    #          error_range=1, agg_mean_style='--')
    plt.legend()
    plt.xlim(right=min(ending_epochs))
    plt.savefig(savepath)
    plt.close()


def plot(ylabel, ploty, plotx, xlabel='Epoch', agged_data=None, sep_data=None, epochs=None, savepath=None, label='mean',
         error_range=1, agg_mean_style='k--'):

    if agged_data is not None:
        if ploty in ['imitation/{}'.format(metric) for metric in
                ['receiver_sample_complexity', 'sample_complexity', 'sender_sample_complexity',
                 'bc_r_loss', 'bc_s_loss', 'sol_r', 'sol_s', 'expert_r_loss', 'expert_s_loss',
                 'imitation_r_acc', 'imitation_s_acc']]:
            agged_data[ploty, 'mean'] = agged_data[ploty, 'mean'].replace(0, np.nan, inplace=False)
            agged_data[ploty, 'std'] = agged_data[ploty, 'std'].replace(0, np.nan, inplace=False)

        try:
            x, y, err = agged_data[plotx], agged_data[ploty, 'mean'], agged_data[ploty, 'std']
        except:
            x, y, err = agged_data[plotx], agged_data[ploty], 0

        y_nonan = [i for i in range(len(y)) if not np.isnan(y[i])]

        if len(y_nonan) != len(y):
            y, x, err = np.array([y[i] for i in y_nonan]), \
                        np.array([x[i] for i in y_nonan]), \
                        np.array([err[i] for i in y_nonan])

        if ploty in ['imitation/{}'.format(metric) for metric in
                     ['bc_r_loss', 'bc_s_loss', 'sol_r', 'sol_s', 'expert_r_loss', 'expert_s_loss',
                      'imitation_r_acc', 'imitation_s_acc']]:
            x, y, err = x[::2], y[::2], err[::2]

        plt.fill_between(x, y - error_range * err, y + error_range * err, alpha=0.2)
        plt.plot(x, y, agg_mean_style, label=label)

    if sep_data is not None:
        for i, run in enumerate(sep_data):
            plt.plot(epochs, run[ploty])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if savepath is not None:
        plt.legend()
        plt.savefig(savepath)


def load_time_series_from_experiment(experiment_path, acc_thres=0.75, last_only=False, n_samples=40):
    # load all interactions
    agged_data, sep_data, last_interaction = {}, {}, {}

    for mode in ['train', 'validation']:
        print(experiment_path)
        print(mode)
        try:
            all_interactions = load_all_interactions(experiment_path, mode=mode, last_only=last_only, n_samples=n_samples)
            agged_data[mode], sep_data[mode], num_seeds = convert_to_acc_df(
                all_interactions,
                acc_thres=acc_thres
            )
            last_interaction[mode] = all_interactions[0]
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

    return agged_data, sep_data, num_seeds, last_interaction


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


def plot_composite(agged_datas, sep_datas, n, args) -> None:
    # load all interactions
    print(args.experiment)
    # agged_data_trains = [agged_datas[experiment] for experiment in args.experiment] # list of dicts.
    # agged_data_trains = [run['train'] for run_list in agged_data_trains for run in run_list]

    for metric in ['context_independence', 'positional_disent', 'bag_of_symbol_disent', 'topographic_sim']:
        plot_means(metric, 'images/training_{}_n_epochs_{}_composite_{}.png'.format(
            metric, args.n_epochs[0], '_'.join(args.experiment)
        ),
                   'compo_metrics/{}'.format(metric), 'epoch', agged_datas, n)

    for metric in ['generalization hold out/', 'uniform holdout/', '']:
        name = metric.split(' ')[0] if len(metric) else metric

        for accuracy in ['acc', 'acc_or']:
            plot_means(metric, 'images/{}_{}_n_epochs_{}_composite_{}.png'.format(
                name, accuracy,args.n_epochs[0], '_'.join(args.experiment)),
                   '{}{}'.format(metric,accuracy), 'epoch', agged_datas, n)

    plot_means('sender entropy', 'images/training_n_epochs_{}_sender_entropy_composite_{}.png'.format(
        args.n_epochs[0], '_'.join(args.experiment)
    ),
               'sender_entropy', 'epoch', agged_datas, n)

    for imitation_metric in ['receiver_sample_complexity', 'sample_complexity', 'sender_sample_complexity',
                             'bc_r_loss', 'bc_s_loss', 'sol_r', 'sol_s', 'expert_r_loss', 'expert_s_loss',
                             'imitation_r_acc', 'imitation_s_acc']:
        plot_means('imitation/{}'.format(imitation_metric),
                   'images/training_n_epochs_{}_{}_composite_{}.png'.format(
                    args.n_epochs[0], imitation_metric, '_'.join(args.experiment)
            ),'imitation/{}'.format(imitation_metric), 'epoch', agged_datas, n)


def plot_metric_wrt_alpha(ylabel, savepath, ploty, agged_data_dict, rs_counts, mode='train'):
    xs = []
    ys = []
    errs = []
    print(list(agged_data_dict.keys()))

    for experiment in agged_data_dict:
        experiment_dict = agged_data_dict[experiment]
        for hyperparam_name in experiment_dict:
            hyperparam_dict = experiment_dict[hyperparam_name]
            df_to_plot = hyperparam_dict[mode]
            num_rs = rs_counts[experiment][hyperparam_name]
            if not 'control' in experiment:
                alpha = float(hyperparam_name.split('im_weight')[1].split('_')[1].split('/')[0])
                if alpha == 1.0: continue
            else:
                alpha = 1e-8
            label = '{};'.format(experiment) + r'$\alpha$' + '={}; n={}'.format(alpha, num_rs)

            last_index = -1
            if ploty in ['imitation/{}'.format(metric) for metric in
                         ['bc_r_loss', 'bc_s_loss', 'sol_r', 'sol_s', 'expert_r_loss', 'expert_s_loss',
                          'imitation_r_acc', 'imitation_s_acc']]:
                last_index = -2

            ys.append(df_to_plot[ploty, 'mean'].iloc[last_index])
            errs.append(df_to_plot[ploty, 'std'].iloc[last_index])
            xs.append(alpha)
            # labels.append(label)

    fig, ax = plt.subplots()
    xs = np.log10(xs)
    ax.scatter(xs, ys)
    ax.errorbar(xs, ys, yerr=errs, fmt="o")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$\log_{10}\alpha$')
    labels = list(range(-8, 2))
    print('labels: ', labels)
    ax.set_xticks(labels)
    labels[0] = r'-$\infty$'
    labels[1] = ''
    ax.set_xticklabels(labels)
    ax.get_xticklabels()[0].set_fontsize(14)

    fig.savefig(savepath)
    plt.close('all')


def composite_plot_metric_wrt_alpha(agged_datas, sep_datas, n, args):
    # y-axis = metric
    # x-axis = log scale of alpha.
    for metric in ['context_independence', 'positional_disent', 'bag_of_symbol_disent', 'topographic_sim']:
        plot_metric_wrt_alpha(metric, 'images/alpha_{}_n_epochs_{}_composite_{}.png'.format(
            metric, args.n_epochs[0], '_'.join(args.experiment)
        ),
                   'compo_metrics/{}'.format(metric), agged_datas, n)

    for metric in ['generalization hold out/', 'uniform holdout/', '']:
        name = metric.split(' ')[0] if len(metric) else metric
        for accuracy in ['acc', 'acc_or']:
            plot_metric_wrt_alpha(metric, 'images/alpha_{}_{}_n_epochs_{}_composite_{}.png'.format(
                name, accuracy,args.n_epochs[0], '_'.join(args.experiment)),
                   '{}{}'.format(metric,accuracy), agged_datas, n)

    plot_metric_wrt_alpha('sender entropy', 'images/alpha_n_epochs_{}_sender_entropy_composite_{}.png'.format(
        args.n_epochs[0], '_'.join(args.experiment)
    ),
               'sender_entropy', agged_datas, n)

    for imitation_metric in ['receiver_sample_complexity', 'sample_complexity', 'sender_sample_complexity',
                             'bc_r_loss', 'bc_s_loss', 'sol_r', 'sol_s', 'expert_r_loss', 'expert_s_loss',
                             'imitation_r_acc', 'imitation_s_acc']:
        plot_metric_wrt_alpha('imitation/{}'.format(imitation_metric),
                   'images/alpha_n_epochs_{}_{}_composite_{}.png'.format(
                    args.n_epochs[0], imitation_metric, '_'.join(args.experiment)
            ),'imitation/{}'.format(imitation_metric), agged_datas, n)



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

def get_data(args):
    agged_datas, sep_datas, n = load_all_time_series(args)
    return agged_datas, sep_datas, n

def data_analysis_suite(args):
    _, sep_datas = load_all_time_series(args)
    sep_data_control, sep_data_experimental = sep_datas['control'][0]['train'], sep_datas['conv'][0]['train']
    percent_runs_improved(sep_data_control, sep_data_experimental)
    means_statistical_test(sep_data_control, sep_data_experimental)
    variances_statistical_test(sep_data_control, sep_data_experimental)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from imitation experiment.')
    parser.add_argument('--runs_path',
                        type=str, default='/ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/imitation/')
    parser.add_argument('--filter',
                        type=bool, default=True)
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['control_simul', 'simul_receiver']
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
                        default=["0.000001", "0.00001", "0.0001", "0.001", "0.01"]#, "0.025", "0.05", "0.075", "0.1", "1.0"]
                        )
    args = parser.parse_args()

    agged_data, sep_datas, n = get_data(args)
    # plot_composite(agged_data, sep_datas, n, args)
    # data_analysis_suite(args)
    composite_plot_metric_wrt_alpha(agged_data, sep_datas, n, args)
    # data_analysis_suite(args)
