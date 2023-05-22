import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.visualize_spread import *

LABELS = {
    'val_acc': 'Validation Accuracy',
    'val_acc_or': 'Per-attribute Validation Accuracy',
    'topsim': 'Topographic Similarity',
    'posdis': 'Positional Disent.', #anglement',
    'bosdis': 'Bag-of-Symbols Disent.',#anglement',
    'ci': 'Context Independence',
    'generalization_acc': 'Zero-shot Generalization Accuracy',
    'generalization_acc_or': 'Per-attribute Zero-shot Gen. Accuracy',
    'uniform_acc': 'Generalization Accuracy',
    'uniform_acc_or': 'Per-attribute Gen. Accuracy',
    'sol_task_loss': 'Speed-of-learning Task Loss',
    'sol_r': 'Speed-of-learning Receiver (Imitation)',
    'sol_s': 'Speed-of-learning Sender (Imitation)',
    'sol_acc_or': 'Speed-of-learning Per-attribute Accuracy (Communication)',
    'sol_acc': 'Speed-of-learning Accuracy (Communication)',
    'epoch_receiver': '# Imitation Epochs until Convergence (Receiver)',
    'epoch_speaker': '# Imitation Epochs until Convergence (Speaker)',
    'mean_loss': 'Imitator Loss on Original Task',
    'r_loss': 'Final Receiver Imitation Loss',
    's_loss': 'Final Sender Imitation Loss'
}


def simple_distribution(y, df):
    """
    Use to visualize distributions of topsim, posdis, bosdis, ci
    generalization acc, zero-shot gen acc, validation acc of experts.
    """
    label = LABELS[y]
    fig, ax = plt.subplots()
    weights = np.ones_like(df[y, '']) / float(len(df[y, '']))

    plt.hist(df[y, ''], alpha=0.5, bins=18, color='green', weights=weights)
    plt.xlabel(label, fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.text(0.8, 0.8,
             r'$\mu$={}'.format("{:.2f}".format(np.mean(df[y,''])) + "\n" +
                                               r'$\sigma$={}'.format("{:.2f}".format(np.std(df[y,''])))),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes,
             fontsize=32
             )
    plt.tight_layout()
    plt.savefig('expert_{}.png'.format(y))
    plt.close()


def build_df_from_perf_log(perf_log:dict, bc_randomseed:int, min_mean_loss:float):
    df = pd.DataFrame(perf_log)
    df['sol_acc'] = speed_of_learning(df['acc'], df['epoch'])
    df['sol_acc_or'] = speed_of_learning(df['acc_or'], df['epoch'])
    df['sol_r'] = speed_of_learning(df['r_acc'], df['epoch'])
    df['sol_s'] = speed_of_learning(df['s_acc'], df['epoch'])
    df['sol_task_loss'] = speed_of_learning(df['mean_loss'] + np.abs(min_mean_loss), df['epoch'])
    df['bc_randomseed'] = bc_randomseed
    df['epoch_speaker'] = df[df['s_acc'] > 0.9].first_valid_index()

    # Drop r_acc, s_acc, acc, acc_or as they all end up being the same.
    # df = df.drop(labels=['acc', 'acc_or', 's_acc', 'r_acc'], axis=1)

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
    # bc_df = bc_df[(np.abs(stats.zscore(bc_df)) < 3).all(axis=1)] # remove outliers
    bc_df['expert_randomseed'] = expert_rs
    bc_df = bc_df.groupby('expert_randomseed').agg(['mean', 'std'])

    return bc_df


def visualize_as_scatter(compo_metric_name, perf_metric, df):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    thres = 5# if 'r_loss' not in perf_metric else 2
    mini_df = df[(np.abs(stats.zscore(df[perf_metric])) < thres)] # remove outliers
    # mini_df = mini_df[df['val_acc_or'] > 0.975]
    x, y = mini_df[compo_metric_name], mini_df[perf_metric]

    print('Doing {} and {}'.format(compo_metric_name, perf_metric))
    normality_test(x)
    normality_test(y)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax_histy.tick_params(axis='both', which='major', labelsize=14)
    ax_histx.tick_params(axis='both', which='major', labelsize=14)

    # now determine nice limits by hand:
    ybinwidth = (np.max(y) - np.min(y)) / 30
    xbinwidth =(np.max(x) - np.min(x)) / 30
    xmax = np.max(x)
    ymax = np.max(y)
    xlim_, xlim = np.min(x), (int(xmax / xbinwidth) + 1) * xbinwidth
    ylim_, ylim = np.min(y), (int(ymax / ybinwidth) + 1) * ybinwidth

    xbins = np.arange(xlim_, xlim + xbinwidth, xbinwidth)
    ybins = np.arange(ylim_, ylim + ybinwidth, ybinwidth)
    ax_histx.hist(x, bins=xbins, alpha=0.6, weights = np.ones_like(x) / float(len(x)))
    ax_histy.text(0.4, 0.8, r'$\mu$={}'.format("{:.2f}".format(np.mean(y)) + "\n" +
                                               r'$\sigma$={}'.format("{:.2f}".format(np.std(y)))),
                  transform=ax_histy.transAxes, fontsize=14)
    ax_histy.hist(y, bins=ybins, orientation='horizontal', alpha=0.6, weights=np.ones_like(y) / float(len(y)))
    ax_histx.text(0.7, 0.7, r'$\mu$={}'.format("{:.2f}".format(np.mean(x))) + "\n" +
            r'$\sigma$={}'.format("{:.2f}".format(np.std(x))), transform=ax_histx.transAxes, fontsize=14)

    # scatterplot and line of best fit
    mini_df.plot.scatter(compo_metric_name, perf_metric, ax=ax, alpha=0.6)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='orange')
    pearsonr, pp = stats.pearsonr(x,y) # note: p-values assume that data are normally distributed.
    spearmanr, sp = stats.spearmanr(x,y)
    ax.text(0.4, 0.8, r'Pearson $R={}, p={}$'.format("{:.2f}".format(pearsonr), "{:.2f}".format(pp))+ '\n' + \
                        r'Spearman $\rho={}, p={}$'.format("{:.2f}".format(spearmanr), "{:.2f}".format(sp)) +
                                         '\nm={}\n'.format("{:.2f}".format(m)), transform=ax.transAxes,
            fontsize=14)
    ax.set_xlabel(LABELS[compo_metric_name], fontsize=18)
    ax.set_ylabel(LABELS[perf_metric[0]], fontsize=18)

    fig.tight_layout()
    fig.savefig('{}_vs_{}.png'.format(LABELS[perf_metric[0]], LABELS[compo_metric_name]))


def plot_curve_over_time(y, df, epochs, label=None, ax=None):

    y, err = df[y, 'mean'], df[y, 'std']


    y = list(y[:len(epochs)])
    err = list(err[:len(epochs)])
    y.extend([float(y[-1])] * 15)
    err.extend([float(err[-1])] * 15)
    y, err = np.array(y), np.array(err)
    epochs = list(epochs)
    epochs.extend([200 * i for i in range(20, 35)])

    ax.fill_between(epochs, y - err, y + err, alpha=0.2)
    ax.plot(epochs, y, label=label)


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


def speed_of_learning(data, epochs, cutoff=50):
    """
    :param data:
    :param epochs:
    :param cutoff: (int) speed of learning up until epoch 10 * i
    :return:
    """
    # integrate under curve up to cutoff
    if cutoff is not None:
        y = [datum if j <= cutoff else 0 for j, datum in enumerate(data)]
    else:
        y = data
    return integrate.cumtrapz(y, x=epochs, initial=0)


if __name__=='__main__':
    results_for_each_expert = []

    min_run_length = float('inf')
    min_mean_loss = float('inf')
    for expert in [10, 21, 29]: # TO EDIT
        try:
            for bc in range(3):
                metadata_bc = load_bc_checkpoints(from_rs=expert, to_rs=bc)
                perf_log = metadata_bc['perf_log']
                min_run_length = min(len(perf_log), min_run_length)
                min_mean_loss = min(0, min(min_mean_loss, min(perf_log['mean_loss'])))
        except:
            continue

    compos = {10: 0.26, 21: 0.36, 29: 0.43}

    for expert in [10, 21, 29]: #range(31): # TO EDIT
        try:
            print('EXPERT', expert)
            for ext in ['', 'rf_with_forcing/']:
                perf_logs = []
                for bc in range(3):
                    metadata_bc = load_bc_checkpoints(from_rs=expert, to_rs=bc, ext=ext)
                    metadata_path = metadata_bc['metadata_path_for_original_model']
                    idx = 1 if 'ccc' not in metadata_path else 7 # reverse compatibility with jean zay
                    metadata_path = '/homedtcl/echeng/EGG/' + '/'.join(metadata_path.split('/')[idx:])
                    metadata_expert, perf_log = load_metadata_from_pkl(metadata_path), \
                                                metadata_bc['perf_log']

                    perf_log = build_df_from_perf_log(perf_log, bc, min_mean_loss)
                    perf_logs.append(perf_log)
                # Round 1 of aggregation for speed of learning: cutoff is min runtime for seeds on each expert.
                for perf_log in perf_logs:
                    # set the last value equal to the min_run_length value for aggregation
                    for colname in ['sol_acc', 'sol_acc_or', 'sol_r', 'sol_s', 'sol_task_loss']:
                        perf_log.at[len(perf_log) - 1, colname] = perf_log.at[min_run_length - 1, colname]
                # entire data
                entire_perf_logs = pd.concat(perf_logs)
                epochs = entire_perf_logs['epoch'].sort_values().unique()
                whole_perf = entire_perf_logs.groupby('epoch').agg(['mean', 'std'])
                label = ('RF ' if len(ext) else 'SV ') + r'$\rho={}$'.format(compos[expert])
                plot_curve_over_time('topsim', whole_perf, epochs, label=label)
            continue

            perf_logs = entire_perf_logs.groupby('bc_randomseed').agg('last')
            perf = get_terminal_bc_stats_for_expert(perf_logs, expert)

            perf['topsim'], perf['bosdis'], perf['posdis'], perf['ci'], perf['val_acc'], perf['val_acc_or'],\
                perf['uniform_acc'], perf['uniform_acc_or'], perf['generalization_acc'], perf['generalization_acc_or'] = \
                metadata_expert['last_validation_compo_metrics']['topographic_sim'], \
                metadata_expert['last_validation_compo_metrics']['bag_of_symbol_disent'], \
                metadata_expert['last_validation_compo_metrics']['positional_disent'], \
                metadata_expert['last_validation_compo_metrics']['context_independence'], \
                metadata_expert['validation_acc'], metadata_expert['validation_acc_or'], \
                metadata_expert['uniformtest_acc'], metadata_expert['uniformtest_acc_or'], \
                metadata_expert['generalization_acc'], metadata_expert['generalization_acc_or']
            results_for_each_expert.append(perf)
        except(FileNotFoundError):
            continue

    plt.legend(fontsize=13)
    plt.ylabel('Topographic Similarity', fontsize=20)
    plt.xlabel('Imitation Training Epochs', fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('test.png')

    df = pd.concat(results_for_each_expert)
    print(df.columns)
    # plot_curve_over_time('s_acc', df)
    input()
    # print(df)
    # print(df.columns)
    # input()

    # for y in ['topsim', 'bosdis', 'posdis', 'ci', 'val_acc', 'generalization_acc', 'uniform_acc']:
    #     simple_distribution(y, df)

    # Correlations and scatterplots
    for x in set(df.columns):
        if 'std' not in x and 'epoch' not in x:# and 'epoch_speaker' not in x:
            for y in ['topsim', 'bosdis', 'posdis', 'ci', 'val_acc_or', 'val_acc']:
                visualize_as_scatter(y, x, df)