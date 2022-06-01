import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.visualize_spread import *



def build_df_from_perf_log(perf_log:dict, bc_randomseed:int, min_mean_loss:float):
    df = pd.DataFrame(perf_log)
    df['sol_acc'] = speed_of_learning(df['acc'], df['epoch'])
    df['sol_acc_or'] = speed_of_learning(df['acc_or'], df['epoch'])
    df['sol_r'] = speed_of_learning(df['r_acc'], df['epoch_receiver'])
    df['sol_s'] = speed_of_learning(df['s_acc'], df['epoch_speaker'])
    df['sol_task_loss'] = speed_of_learning(df['mean_loss'] + np.abs(min_mean_loss), df['epoch'])
    df['bc_randomseed'] = bc_randomseed

    # Drop r_acc, s_acc, acc, acc_or as they all end up being the same.
    df = df.drop(labels=['acc', 'acc_or', 's_acc', 'r_acc'], axis=1)

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
    # print('terminal bc stats for expert')
    # print(bc_df)
    bc_df = bc_df[(np.abs(stats.zscore(bc_df)) < 3).all(axis=1)] # remove outliers
    bc_df['expert_randomseed'] = expert_rs
    bc_df = bc_df.groupby('expert_randomseed').agg(['mean', 'std'])

    return bc_df


def visualize_as_scatter(compo_metric_name, perf_metric, df):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    thres = 3 if 'r_loss' not in perf_metric else 2
    mini_df = df[(np.abs(stats.zscore(df[perf_metric])) < thres)] # remove outliers
    x, y = mini_df[compo_metric_name], mini_df[perf_metric]

    print('Doing {} and {}'.format(compo_metric_name, perf_metric))
    # normality_test(x)
    # normality_test(y)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # now determine nice limits by hand:
    ybinwidth = (np.max(y) - np.min(y)) / 30
    xbinwidth =(np.max(x) - np.min(x)) / 30
    xmax = np.max(x)
    ymax = np.max(y)
    xlim_, xlim = np.min(x), (int(xmax / xbinwidth) + 1) * xbinwidth
    ylim_, ylim = np.min(y), (int(ymax / ybinwidth) + 1) * ybinwidth

    xbins = np.arange(xlim_, xlim + xbinwidth, xbinwidth)
    ybins = np.arange(ylim_, ylim + ybinwidth, ybinwidth)
    ax_histx.hist(x, bins=xbins)
    ax_histy.text(0.4, 0.8, r'$\mu$={}'.format("{:.2f}".format(np.mean(y)) + "\n" +
                                               r'$\sigma$={}'.format("{:.2f}".format(np.std(y)))),
                  transform=ax_histy.transAxes)
    ax_histy.hist(y, bins=ybins, orientation='horizontal')
    ax_histx.text(0.7, 0.7, r'$\mu$={}'.format("{:.2f}".format(np.mean(x))) + "\n" +
            r'$\sigma$={}'.format("{:.2f}".format(np.std(x))), transform=ax_histx.transAxes)

    # scatterplot and line of best fit
    mini_df.plot.scatter(compo_metric_name, perf_metric, ax=ax)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='orange')
    pearsonr, pp = stats.pearsonr(x,y) # note: p-values assume that data are normally distributed.
    spearmanr, sp = stats.spearmanr(x,y)
    ax.text(0.6, 0.8, 'Pearson R={}, p={}\n'.format("{:.2f}".format(pearsonr), "{:.2f}".format(pp))+
                        'Spearman R={}, p={}\n'.format("{:.2f}".format(spearmanr), "{:.2f}".format(sp)) +
                                         'm={}\n'.format("{:.2f}".format(m)), transform=ax.transAxes)
    fig.savefig('{}_vs_{}.png'.format(perf_metric, compo_metric_name))


def normality_test(data, alpha=1e-3):
    stat, p = stats.normaltest(data)

    print('Dagostino K2 test')
    if p < alpha:
        print('Probably not normal')
    else:
        print('Cannot reject that data comes from normal distribution')

    print('Shapiro-wilk')
    stat, p = stats.shapiro(data)
    if p < alpha:
        print('Probably not normal')
    else:
        print('Cannot reject that data comes from normal distribution')


def speed_of_learning(data, epochs):
    # integrate under curve
    return integrate.cumtrapz(data, x=epochs, initial=0)


if __name__=='__main__':
    results_for_each_expert = []

    min_run_length = float('inf')
    min_mean_loss = float('inf')
    for expert in range(100):
        try:
            for bc in range(101, 131):
                metadata_bc = load_bc_checkpoints(from_rs=expert, to_rs=bc)
                perf_log = metadata_bc['perf_log']
                min_run_length = min(len(perf_log), min_run_length)
                min_mean_loss = min(0, min(min_mean_loss, min(perf_log['mean_loss'])))
        except:
            continue

    for expert in range(100):
        try:
            perf_logs = []
            for bc in range(101, 131):
                    metadata_bc = load_bc_checkpoints(from_rs=expert, to_rs=bc)
                    metadata_path = metadata_bc['metadata_path_for_original_model'].split('/')
                    metadata_path = '/'.join([metadata_path[0], 'n_val_10_n_att_2_vocab_100_max_len_3_hidden_500/', metadata_path[1]])
                    metadata_expert, perf_log = load_metadata_from_pkl(metadata_path), \
                                                metadata_bc['perf_log']
                    perf_log = build_df_from_perf_log(perf_log, bc, min_mean_loss)
                    perf_logs.append(perf_log)

            # Round 1 of aggregation for speed of learning: cutoff is min runtime for seeds on each expert.
            for perf_log in perf_logs:
                # set the last value equal to the min_run_length value for aggregation
                for colname in ['sol_acc', 'sol_acc_or', 'sol_r', 'sol_s', 'sol_task_loss']:
                    perf_log.at[len(perf_log) - 1, colname] =  perf_log.at[min_run_length - 1, colname]

            perf_logs = pd.concat(perf_logs).groupby('bc_randomseed').agg('last')
            perf = get_terminal_bc_stats_for_expert(perf_logs, expert)

            perf['topsim'], perf['bosdis'], perf['posdis'], perf['ci'] = \
                metadata_expert['last_validation_compo_metrics']['topographic_sim'], \
                metadata_expert['last_validation_compo_metrics']['bag_of_symbol_disent'], \
                metadata_expert['last_validation_compo_metrics']['positional_disent'], \
                metadata_expert['last_validation_compo_metrics']['context_independence']
            results_for_each_expert.append(perf)
        except(FileNotFoundError):
            continue

    df = pd.concat(results_for_each_expert)

    # Correlations and scatterplots
    # for x in (set(df.columns) - set((('topsim',""), ('bosdis',""), ('posdis',""), ('ci',"")))):
    #     if 'std' not in x:
    #         for y in ['topsim', 'bosdis', 'posdis', 'ci']:
    #             visualize_as_scatter(y, x, df)

    # Plot distributions for metrics:
    for y in ['topsim', 'bosdis', 'posdis', 'ci']:
        plot_hist(df, y)