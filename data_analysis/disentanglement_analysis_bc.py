import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from experiments.loader import *


def build_df_from_perf_log(perf_log:dict, bc_randomseed:int):
    df = pd.DataFrame(perf_log)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] # remove outliers
    df['bc_randomseed'] = bc_randomseed

    return df


def get_terminal_bc_stats_for_expert(bc_df:pd.DataFrame, expert_rs:int):
    """
    Gives (avg, std) stats on terminal state of all BC runs for a given expert for
    1. # of iterations until convergence
    2. validation accuracy, accuracy_or on original task, mean_loss
    3. validation loss from training
    :param bc_df: concated df of all behavioral cloning runs for one expert.
    :return: all stats
    """
    bc_df = bc_df[(np.abs(stats.zscore(bc_df)) < 2).all(axis=1)] # remove outliers
    bc_df['expert_randomseed'] = expert_rs
    bc_df = bc_df.groupby('expert_randomseed').agg(['mean', 'std'])

    return bc_df


def visualize_as_scatter(compo_metric_name, perf_metric, df):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    print('perf metric: ', perf_metric)
    thres = 3 if 'r_loss' not in perf_metric else 1
    mini_df = df[(np.abs(stats.zscore(df[perf_metric])) < thres)] # remove outliers
    x, y = mini_df[compo_metric_name], mini_df[perf_metric]

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # now determine nice limits by hand:
    ybinwidth = (np.max(y) - np.min(y)) / 30
    xbinwidth =(np.max(x) - np.min(x)) / 30
    xmax = np.max(np.abs(x))
    ymax = np.max(np.abs(y))
    xlim_, xlim = np.min(x), (int(xmax / xbinwidth) + 1) * xbinwidth
    ylim_, ylim = np.min(y), (int(ymax / ybinwidth) + 1) * ybinwidth

    xbins = np.arange(xlim_, xlim + xbinwidth, xbinwidth)
    ybins = np.arange(ylim_, ylim + ybinwidth, ybinwidth)
    ax_histx.hist(x, bins=xbins)
    ax_histy.hist(y, bins=ybins, orientation='horizontal')

    # scatterplot and line of best fit
    mini_df.plot.scatter(compo_metric_name, perf_metric, ax=ax)

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='orange')
    ax.text(0.8 * max(x), 0.8 * max(y), 'Spearman R={}\n'.format("{:.2f}".format(np.corrcoef(x,y)[0][1])) +
                                         'm={}\n'.format("{:.2f}".format(m)))
    fig.savefig('{}_vs_{}.png'.format(perf_metric, compo_metric_name))


if __name__=='__main__':
    results_for_each_expert = []

    for expert in range(100):
        try:
            perf_logs = []
            for bc in range(101, 123):
                    metadata_bc = load_bc_checkpoints(from_rs=expert, to_rs=bc)
                    metadata_expert, perf_log = load_metadata_from_pkl(metadata_bc['metadata_path_for_original_model']), \
                                                metadata_bc['perf_log']
                    perf_logs.append(build_df_from_perf_log(perf_log, bc))
            perf_logs = pd.concat(perf_logs).groupby('bc_randomseed').agg('last')
            # print('raw data for expert {}: {}'.format(expert, perf_logs))
            perf = get_terminal_bc_stats_for_expert(perf_logs, expert)
            perf['topsim'], perf['bosdis'], perf['posdis'] = \
                metadata_expert['last_validation_compo_metrics']['topographic_sim'], \
                metadata_expert['last_validation_compo_metrics']['bag_of_symbol_disent'], \
                metadata_expert['last_validation_compo_metrics']['positional_disent']
            # print("perf for expert {}: {}".format(expert, perf))
            results_for_each_expert.append(perf)
        except(FileNotFoundError):
            continue

    df = pd.concat(results_for_each_expert)
    print(df)

    for x in (set(df.columns) - set((('topsim',""), ('bosdis',""), ('posdis',"")))):
        if 'std' not in x:
            for y in ['topsim', 'bosdis', 'posdis']:
                visualize_as_scatter(y, x, df)