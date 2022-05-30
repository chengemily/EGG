import matplotlib.pyplot as plt
import argparse
import torch
from pathlib import Path
import typing
from typing import TypeVar, Generic, Sequence, Iterable
import pandas as pd
import numpy as np
import seaborn as sns
from egg.zoo.addition_game.analysis_util import *
from scipy.stats import spearmanr, pearsonr


def success_v_number_symbols(args):
    fig, ax = plt.subplots()

    for experiment in args.experiment:
        print('\nEXPERIMENT: ', experiment)
        all_interactions, _ = load_all_interactions(args.runs_path + experiment, mode='validation')
        if args.filter:
            all_interactions, _ = filter_interactions(all_interactions)

        best_vals = [np.max([torch.mean(epoch.aux['acc']) for epoch in rs]) for rs in all_interactions]
        best_val_messages, best_val_epochs = get_best_val_messages(all_interactions)
        input_size = best_val_messages[0].shape[0]

        if args.holdout:
            holdout_interactions, _ = load_all_interactions(args.runs_path + experiment, mode='uniform')
            holdout_messages = get_best_messages_based_on_val(holdout_interactions, best_val_epochs)
            holdout_scores = [rs['acc'].mean() for rs in holdout_messages]
            print('holdout scores: ', holdout_scores)

        # Define symbol density, synonymy, polysemy.
        num_unique_symbols_per_input = [len(rs['symbol'].unique()) / input_size for rs in best_val_messages]
        synonymy = [rs.groupby('sum').agg({'symbol': ['count', pd.Series.nunique]}) for rs in best_val_messages]
        synonymy = [np.mean(rs['symbol', 'nunique'] / rs['symbol', 'count']) for rs in synonymy]
        polysemy = [
            np.mean(rs.groupby('symbol').agg(pd.Series.nunique)['sum'])
            for rs in best_val_messages
        ]
        print('Symbol density: ', [np.mean(num_unique_symbols_per_input), np.std(num_unique_symbols_per_input)])
        print('Synonymy: ', [np.mean(synonymy), np.std(synonymy)])
        print('Polysemy: ', [np.mean(polysemy), np.std(polysemy)])
        print('Spearman: ')
        print('Symbol density vs best validation accuracy')
        print(spearmanr(num_unique_symbols_per_input, best_vals))
        print('Synonymy vs best validation accuracy')
        print(spearmanr(synonymy, best_vals))
        print('Polysemy vs best validation accuracy')
        print(spearmanr(polysemy, best_vals))
        print(spearmanr(synonymy, num_unique_symbols_per_input))

        if args.holdout:
            print('Symbol density vs best holdout accuracy')
            print(spearmanr(num_unique_symbols_per_input, holdout_scores))
            print('Synonymy vs best holdout accuracy')
            print(spearmanr(synonymy, holdout_scores))
            print('Polysemy vs best holdout accuracy')
            print(spearmanr(polysemy, holdout_scores))

        print('Pearson')
        print('Symbol density vs best validation accuracy')
        print(pearsonr(num_unique_symbols_per_input, best_vals))
        print('Synonymy vs best validation accuracy')
        print(pearsonr(synonymy, best_vals))
        print('Polysemy vs best validation accuracy')
        print(pearsonr(polysemy, best_vals))

        m, b = np.polyfit(num_unique_symbols_per_input, best_vals, 1)
        label = experiment.split('_')[-1] if experiment != 'full_6000' else '1.0'
        ax.scatter(
            synonymy,
            best_vals,
            label=r'$ \beta $'+ f'$ = {{{label}}}$' + ": y={:.2f}x + {:.2f}".format(m, b)
        )
    ax.set_ylabel('Validation accuracy')
    ax.set_xlabel('Symbol density')
    ax.legend()
    fig.savefig('images/success_v_num_symbols.png')


def plot_symbols(interaction_df, savepath):
    pivot_df = interaction_df.pivot(index='x', columns='y', values='symbol').fillna(-1).astype(int)
    sns.heatmap(pivot_df, annot=True, fmt='d', annot_kws={'size':5}, mask=pivot_df == -1)
    plt.xlim(15, 20)
    plt.ylim(12, 17)
    plt.xlabel(f'$x_1$')
    plt.ylabel(f'$x_2$')
    plt.savefig(savepath)
    plt.close()


def symbol_analysis(args):
    for experiment in args.experiment:
        for mode in ['validation', 'uniform', 'permutation']:
            all_interactions, epochs = load_all_interactions(args.runs_path + experiment, mode=mode)
            good_acc_interactions = None

            if mode == 'validation':
                good_acc_interactions, best_runs = filter_interactions(all_interactions, acc_thres=0 if not args.filter else 0.75)
                good_acc_interactions, best_val_epochs = get_best_val_messages(good_acc_interactions)
            else:
                good_acc_interactions = filter_interactions_on_runs(all_interactions, best_runs)
                good_acc_interactions = get_best_messages_based_on_val(good_acc_interactions, best_val_epochs)

            if good_acc_interactions is not None:
                for i, interaction_df in enumerate(good_acc_interactions):

                    # Number of unique symbols
                    print('Number of unique symbols: ', len(interaction_df['symbol'].unique()))

                    # Plot symbols over input space as heatmap
                    plot_symbols(interaction_df, './images/addition_{}_symbols_{}_{}.png'.format(experiment, mode, i))

                polysemy(good_acc_interactions)


def polysemy(interaction_dfs):
    results = {
        'avg # inputs per symbol':[],
        'avg # sums per symbol': [],
        '% polysemes': [],
        'median dist': [],
        'max dist': [],
        'avg dist': [],
        'median L1 dist': [],
        'max L1 dist': [],
        'avg L1 dist': [],
        'min L1 dist': []
    }

    for interaction_df in interaction_dfs:
        symbol_map = interaction_df.groupby('symbol')

        # Quantify polysemy
        polysemy_stats = symbol_map.agg({'acc': 'mean', 'output': ['count', pd.Series.nunique], 'sum': ['count', pd.Series.nunique]})
        results['avg # inputs per symbol'].append(np.mean(polysemy_stats['sum', 'count']))
        results['avg # sums per symbol'].append(np.mean(polysemy_stats['sum', 'nunique']))

        # Extract failures
        failed_symbol_map = [group for _, group in symbol_map if group['acc'].mean() < 1]
        if len(failed_symbol_map):
            failed_symbol_map = pd.concat(failed_symbol_map).groupby('symbol')
            recvr_failures = 0
            sender_failures = 0
            for name, group in failed_symbol_map:
                num_sender_failures, correct = 0, []
                if group['acc'].mean() > 0:
                    # get target sum
                    correct = group[group['acc'] == 1.0]
                    target_sum = correct['sum'].iloc[0]
                    num_sender_failures = len(group[group['sum'] != target_sum])
                    sender_failures += num_sender_failures
                recvr_failures += len(group) - num_sender_failures - len(correct)

            print('Percent of failures due to sender polysemy: ', sender_failures / (recvr_failures + sender_failures))
            print('Percent of failures due to receiver error: ', recvr_failures / (recvr_failures + sender_failures))
        else:
            print('No failures')

        # L1 distances between input pairs
        deduped_symbol_map = [group for _, group in symbol_map if len(group) > 1]
        if len(deduped_symbol_map):
            interaction_results = {
                'median L1 dist': [],
                'max L1 dist': [],
                'avg L1 dist': [],
                'min L1 dist': []
            }

            deduped_symbol_map = pd.concat(deduped_symbol_map).groupby('symbol')
            for name, group in deduped_symbol_map:
                norms = []
                for i in range(len(group) - 1):
                    for j in range(i + 1, len(group)):
                        coord_1, coord_2 = group.iloc[i][['x', 'y']], group.iloc[j][['x', 'y']]
                        l1_norm = np.linalg.norm(coord_2 - coord_1, ord=1)
                        norms.append(l1_norm)

                # produce max L1, min L1, median L1, avg L1.
                interaction_results['median L1 dist'].append(np.median(norms))
                interaction_results['max L1 dist'].append(np.max(norms))
                interaction_results['min L1 dist'].append(np.min(norms))
                interaction_results['avg L1 dist'].append(np.mean(norms))

            for item in ['median L1 dist', 'max L1 dist', 'min L1 dist', 'avg L1 dist']:
                results[item].append(np.mean(interaction_results[item]))

            ranges = deduped_symbol_map.agg('max') - deduped_symbol_map.agg('min')

            for factor in ['sum']: #, 'L1_inputs':
                results['median dist'].append(np.median(ranges[factor]))
                results['max dist'].append(np.max(ranges[factor]))
                results['avg dist'].append(np.mean(ranges[factor]))

            results['% polysemes'].append(len(ranges) / len(symbol_map))
        else:
            # bijection between symbols and input pairs
            for factor in ['median dist', 'max dist', 'avg dist'] + \
                    ['median L1 dist', 'max L1 dist', 'min L1 dist', 'avg L1 dist', '% polysemes']:
                results[factor].append(0)

    # Print aggregate results on median sum range, L1 range, % polysemes, etc.
    results = pd.DataFrame(results)
    print(results.mean(axis=0))
    print(results.std(axis=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from addition experiment.')
    parser.add_argument('--runs_path',
                        type=str, default='./additions_interactions/')
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['full_6000']
                        )
    parser.add_argument('--holdout', type=bool, default=True)
    parser.add_argument('--filter', type=bool)

    args = parser.parse_args()
    symbol_analysis(args)
    success_v_number_symbols(args)