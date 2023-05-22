import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from egg.zoo.imitation_learning.loader import *
from egg.zoo.imitation_learning.visualize_spread import *
from egg.zoo.imitation_learning.analysis.imitation_analysis import *
from egg.zoo.imitation_learning.analysis.bc_analysis import plot_curve_over_time

from egg.core.language_analysis import *
from scipy.stats import skew, entropy, kurtosis

YLIMS = {
    'sender':{
        2: 1.0, 3: 0.8, 4: 0.55, 5: 0.45, 10: 0.3, 20: 0.9, 30: 0.9
    },
    'receiver':{
        2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0
    }
    }
LABELS = {
        'topographic_sim': r'Expert Topsim',
        'positional_disent': r'Expert Positional Disentanglement',
        'bag_of_symbol_disent': r'Expert Bag-of-Symbols Disentanglement',
        'context_independence': r'Expert Context Independence',
        'validation_accuracy': r'Expert Validation Accuracy',
        'acc': r'Learned Expert Language Weights',
        'weight': r'Learned Expert Language Weights',
        # 'weight': r'Receiver Validation Accuracy',
        'train_acc': r'Training Accuracy',
        'edit_distance': r'Edit Distance',
        'validation_acc': r'Expert Validation Accuracy',
        'uniformtest_acc': r'Expert Generalization Accuracy',
        'generalization_acc': r'Expert Zero-Shot Generalization Accuracy'
    }


def estimate_skew(compos, weights):
    # Skewness of compos:
#     print('Skewness of topsim: ', skew(compos))

    # Generate new distribution of experts based on weights
    dist = []
    for i, compo in enumerate(compos):
        dist.extend([i] * int(weights[i]*1000))

    skew_weights = skew(dist)
    return skew(compos), skew_weights


def plot_histogram(compo_metric, metric, metadatas, dataset, args, experts, ax=None, is_train=False, save=False):
    # Get data
    xs, ys, devs = [], [], []

    for i, expert in enumerate(experts):
        expert_metadata = metadatas[i]
        xs.append(expert_metadata[compo_metric])
        ys.append(float(dataset['{}_{}'.format(expert, metric), 'mean']))
        devs.append(float(dataset['{}_{}'.format(expert, metric), 'std']))
    data = pd.DataFrame.from_dict(
        {compo_metric: xs,
         'Experts': experts,
         metric: ys,
         'sd': devs
         })
    data = data.sort_values(compo_metric)

    print(data)

    # Plot data
    if ax is None:
        fig, ax = plt.subplots()

    x_pos = np.arange(args.experts)
    ax.bar(x_pos, data[metric],
           yerr=data['sd'],
            align='center', alpha=0.5, ecolor='black', capsize=10)

    if is_train: metric = 'train_' + metric
    # ax.set_ylabel(LABELS[metric])
    # ax.set_xlabel(LABELS[compo_metric])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["%.2f" % ts for ts in data[compo_metric]])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    ax.set_ylim(top=YLIMS[args.agent][args.experts])
    ax.yaxis.grid(True)

    title = (r'$RF$ ' if int(args.rf) else r'$Supervision$ ') + \
            (r'($\lambda={}$)'.format(args.ent) if int(args.rf) else "")
    if args.experts == 2:
        ax.set_title(title, fontsize=20)
    if args.rf == 1 and args.ent == 0:
        ax.set_ylabel(r'$k={}$'.format(args.experts), fontsize=20)

    # Save the figure and show
    if save:
        plt.tight_layout()
        plt.savefig('{}_vs_{}_ent_{}_n_{}_rf_{}.png'.format(metric, compo_metric, args.ent, args.experts, args.rf))


def compute_expert_weights(expert_list, interactions, compos, args):
    if args.agent == 'sender':
        return compute_sender_weights(expert_list, interactions, compos)
    elif args.agent == 'receiver':
        return compute_receiver_weights(expert_list, interactions, compos)


def compute_receiver_weights(expert_list, interactions, compos):
    all_rs_weights = []
    all_rs_entropies = []
    all_rs_skews = []

    for i, interaction in enumerate(interactions):
        interaction = interaction[-1]
        out_accuracies = [interaction.aux['{}_acc'.format(expert)].item() for expert in expert_list]
        all_rs_weights.append(out_accuracies)
        all_rs_entropies.append(entropy(out_accuracies, base=2))
        all_rs_skews.append(estimate_skew(compos, out_accuracies)[-1])

    all_rs_weights = np.array(all_rs_weights)
    mean_weights = np.mean(all_rs_weights, axis=0)
    std_weights = np.std(all_rs_weights, axis=0)
    mean_entropy, std_entropy = np.mean(all_rs_entropies), np.std(all_rs_entropies)
    mean_skew, std_skew = np.mean(all_rs_skews), np.std(all_rs_skews)

    return mean_weights, std_weights, mean_entropy, std_entropy, mean_skew, std_skew


def compute_sender_weights(expert_list, interactions, compos):
    # expert_list: [10, 3, 29] e.g.
    # interactions: list of interactions (across random seeds)
    # compos: [0.25, 0.32, 0.43] list of compos corresponding to expert_list
    ref_expert = expert_list[0]

    all_rs_weights = []
    all_rs_entropies = []
    all_rs_skews = []
    print('Expert list: ', expert_list)
    print('Compos: ', compos)
    expert_list = [expert for _, expert in sorted(zip(compos, expert_list))]
    compos.sort()
    print('Expert list sorted by compos: ', expert_list)
    # Sort expert list by compo

    for i, interaction in enumerate(interactions):
        interaction = interaction[-1]
        print(interaction)
        try:
            # print(interaction)
            imitator_message = interaction.aux['{}_imitator_message'.format(ref_expert)]
            expert_messages = [interaction.aux['{}_expert_message'.format(expert)] for expert in expert_list]
        except:
            continue

        # batch size x messages -> batch_size x 1
        batch_size = imitator_message.shape[0]
        msg_accuracies = [(imitator_message == expert_message).float().mean(dim=-1) for expert_message in expert_messages]

        # add random baseline
        msg_accuracies.append(torch.ones(batch_size) * 0.1)#max(0.1, 1/(len(expert_list)+1))) # random chance

        # put them together and argmax
        msg_accuracies = torch.stack(msg_accuracies, dim=0)
        max_accuracies, winners = torch.max(msg_accuracies, dim=0)
        counts = torch.bincount(winners, minlength=len(expert_list) + 1)
        weights = counts / torch.sum(counts)
        expert_only_weights = counts[:-1] / torch.sum(counts[:-1])
        print('expert only weights:', expert_only_weights)

        all_rs_weights.append(weights)
        all_rs_entropies.append(
            # np.var(expert_only_weights)
            entropy(expert_only_weights, base=2)
        )
        all_rs_skews.append(estimate_skew(compos, expert_only_weights)[-1])

    all_rs_weights = torch.stack(all_rs_weights, dim=0)
    mean_weights = torch.mean(all_rs_weights, dim=0)
    std_weights = torch.std(all_rs_weights, dim=0)

    mean_entropy, std_entropy = np.mean(all_rs_entropies), np.std(all_rs_entropies)
    mean_skew, std_skew = np.mean(all_rs_skews), np.std(all_rs_skews)

    return mean_weights.numpy(), std_weights.numpy(), mean_entropy, std_entropy, mean_skew, std_skew


def inspect_messages(val, tr):
    k=10
    print('Validation imitator message')
    print(val.aux['10_imitator_message'][:k].numpy())
    print("--------------------------------------------------------")
    print('Noncompo expert message')
    print(tr.aux['10_expert_message'][:k].numpy())
    print('Noncompo imitator message')
    print(tr.aux['10_imitator_message'][:k].numpy())
    print('Symbol probs')
    print(tr.aux['10_class_probs'][:k].numpy())
    print("--------------------------------------------------------")
    print('Compo expert message')
    print(tr.aux['29_expert_message'][:k].numpy())
    print('Compo imitator message')
    print(tr.aux['29_imitator_message'][:k].numpy())
    print('Symbol probs')
    print(tr.aux['29_class_probs'][:k].numpy())
    return


def load_data(args, rf, n, ent, dv=False, last_only=False, n_samples=40, skew=0, config_file=None):
    ###### Get experts
    experts_sorted_by_topsim = np.array([10, 3, 24, 2, 4, 25, 26, 20, 5, 7, 17, 15, 19, 14, 21, 8, 9, 28,
                                         11, 0, 13, 22, 1, 18, 6, 12, 23, 16, 27, 29])
    if args.skew:
        # Load in json config file
        if config_file is None:
            if n == 2:
                filepath = '/homedtcl/echeng/EGG/bash_scripts/mixture_senders/expert_distributions_config.json'
            else:
                filepath = '/homedtcl/echeng/EGG/bash_scripts/mixture_senders/vars_expert_distributions_config.json'
        else:
            filepath = config_file
        with open(filepath) as json_file:
            expert_config = json.load(json_file)
            idx = np.array(expert_config[str(n)]['experts'][skew]).astype(int)
    else:
        experts_sorted_by_topsim = np.array([10, 3, 24, 2, 4, 25, 26, 20, 5, 7, 17, 15, 19, 14, 21, 8, 9, 28,
                                         11, 0, 13, 22, 1, 18, 6, 12, 23, 16, 27, 29])
        idx = np.round(np.linspace(0, len(experts_sorted_by_topsim) - 1, int(n))).astype(int)

    experts = list(experts_sorted_by_topsim[idx])
    print('found experts:', experts)

    # Load expert data
    metadatas = []
    for expert in experts:
        metadata = load_metadata_from_pkl(
            '/homedtcl/echeng/EGG/checkpoints/basic_correlations/saved_models/checkpoint_wrapper_randomseed{}.pkl'.format(
                expert))
        for compo_metric in metadata['last_validation_compo_metrics']:
            metadata[compo_metric] = metadata['last_validation_compo_metrics'][compo_metric]
        # print(metadata)
        metadatas.append(metadata)
    compos = [metadata['topographic_sim'] for metadata in metadatas]

    ######### Load imitator performance
    path = args.runs_path + '{}{}/{}rf_{}/experts_{}/ent_{}/'.format(
        'receiver/' if args.agent == 'receiver' else '',
        'disjoint_vocab' if dv and rf else "",
        # 'skew_1/',
        '',
        rf, n, ent
    )
    agged_data, sep_data, num_seeds, last_interactions = load_time_series_from_experiment(
        path, acc_thres=-1, last_only=last_only, n_samples=n_samples)
    # val = last_interactions['validation'][0][0]
    # tr = last_interactions['train'][0][0]
    # inspect_messages(val, tr)

    train, validation = agged_data['train'], agged_data['validation']

    # Add weights + distributional characteristics to the data
    if args.agent == 'sender':
        mean_weights, std_weights, mean_weight_ent, std_weight_ent, mean_weight_skew, std_weight_skew = \
        compute_expert_weights(experts, last_interactions['validation'], compos, args,)

        for i, expert in enumerate(experts):
            validation['{}_weight'.format(expert), 'mean'] = mean_weights[i]
            validation['{}_weight'.format(expert), 'std'] = std_weights[i]
        validation['weight_skew', 'mean'], validation['weight_skew', 'std'] = mean_weight_skew, std_weight_skew
        validation['weight_entropy', 'mean'], validation['weight_entropy', 'std'] = mean_weight_ent, std_weight_ent
    elif args.agent == 'receiver':
        mean_weights, std_weights, mean_weight_ent, std_weight_ent, mean_weight_skew, std_weight_skew = \
            compute_expert_weights(experts, last_interactions['validation'], compos, args, )

        for i, expert in enumerate(experts):
            validation['{}_weight'.format(expert), 'mean'] = mean_weights[i]
            validation['{}_weight'.format(expert), 'std'] = std_weights[i]
        validation['weight_skew', 'mean'], validation['weight_skew', 'std'] = mean_weight_skew, std_weight_skew
        validation['weight_entropy', 'mean'], validation['weight_entropy', 'std'] = mean_weight_ent, std_weight_ent

    return train, validation, experts, metadatas, last_interactions


def print_validation_compo(validation, experts, metadatas):
    ref = experts[0]

    for compo in ['topographic_sim',
        'positional_disent',
        'bag_of_symbol_disent',
        'context_independence']:
        name = '{}_{}'.format(ref, compo)

        # Get average compo from metadatas:
        compos = np.array([metadata[compo] for metadata in metadatas])
        print('Avg compo: ', np.mean(compos))

        # Get resulting compo
        print('Result {}: '.format(compo))
        print(float(validation[name, 'mean']))


def plot_training_curves(to_plot: str, experts, df, epochs, metadatas, compo_metric, title="", save=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    print('ax: ', ax)

    for i, expert in enumerate(experts):
        plot_curve_over_time('{}_{}'.format(expert, to_plot), df, epochs, label='{}={}'.format(
            LABELS[compo_metric],
            "%.2f" % (metadatas[i][compo_metric]),
        ), ax=ax)

    ax.set_ylim(top=1.0)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_title(title, fontsize=22)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    if save:
        plt.legend(fontsize=18)
        plt.ylabel(title, fontsize=22)
        plt.xlabel('Imitation Training Epochs', fontsize=22)
        plt.tight_layout()
        plt.savefig('{}.png'.format(title))
        plt.close()

def plot_SEs(xs, ys, errs, expert_list, args, title="Learned Expert Weight Entropy", fig=None, ax=None, left=False):
    # xs: list of lambdas
    # ys: list of list of skews or entropies
    # expert list: [2, 3, 4] experts

    # Plots skewness vs entropy
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5,6))

    x_pos = np.arange(len(ys[0]))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["%.2f" % x for x in xs] + ['SV'])
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, expert_ys in enumerate(ys):
        expert_errs = errs[i]
        ax.plot(x_pos, expert_ys, '-', alpha=0.5,
                color=colors[2],
                # color='g' if not i else 'r',
                linewidth=5, label=r'$k={}$'.format(expert_list[i]))
        ax.errorbar(x_pos, expert_ys, yerr=expert_errs, fmt="o", markersize=10,
                    # color='g' if not i else 'r',
                    color = colors[2],
                    # color=ax.get_lines()[-1].get_c(),
                    alpha=0.6)

    # plt.xlabel(r'Entropy Regularization Coeff. ($\lambda$)', fontsize=20)
    if not left:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel(title, rotation=-90, fontsize=18, labelpad=18)
    else:
        ax.set_ylabel(title, fontsize=18)

    # plt.legend(fontsize=18, loc='lower right')
    # plt.tight_layout()
    # plt.savefig(args.savepath)
    # plt.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from imitation experiment.')
    parser.add_argument('--runs_path',
                        type=str, default='/homedtcl/echeng/EGG/checkpoints/mixture/')
    parser.add_argument('--rf',
                        default=1,
                        )
    parser.add_argument(
        '--agent', default='sender', choices=['sender', 'receiver']
    )
    parser.add_argument('--experts',
                        type=int,
                        default=2)
    parser.add_argument('--ent',
                        default=0.01
                        )
    parser.add_argument('--skew', default=0)
    parser.add_argument('--dv', default=False)
    parser.add_argument('--savepath', type=str)
    args = parser.parse_args()

    # Define subplots
    last_only = True
    expert_list = [4]#, 4, 5]
    # ents = [0.0]
    # skews = [0, 2, 4, 6, 8]
    # expert_list = [3]
    # ents = [0.0]
    ents = [0.0, 0.01, 0.1, 1.0]
    # expert_list = [2]
    # ents = [1.0]

    # fig, axs = plt.subplots(len(expert_list), len(ents) + 1, sharey='row', figsize=(7.5, 3.25 * len(expert_list))) #3.25 * len(expert_list)))
    # fig, axs = plt.subplots()#figsize=(3, 3))
    compo_metric, metric = 'topographic_sim', 'weight'
    # skew_or_entropy = 'entropy'
    # fig, axs = plt.subplots(len(expert_list), len(ents) + 1, sharey='row', figsize=(9, 2.6 * len(expert_list))) #3.25 * len(expert_list)))

    # Test supervision
    all_skews, all_skew_errs = [], []
    all_entropies, all_entropy_errs = [], []
    for i, n in enumerate(expert_list):
        args.experts = n

        # reinforce
        args.rf = 1

        skews, skew_errs = [], []
        entropies, entropy_errs = [], []
        for j, ent in enumerate(ents):
            # if ent not in all_data: all_data[n][ent] = {}
            args.ent = ent
            print('rf: {}, ent: {}'.format(args.rf, ent))
            train, validation, experts, metadatas, last_interaction = load_data(args, args.rf, n, ent, args.dv, last_only=last_only)
            # plot_training_curves('loss', experts, train, train['epoch'], metadatas, compo_metric, 'Training Loss {}'.format(
            # ('(RF)' if args.rf else '(SV)')))
            # plot_training_curves('acc', experts, train, train['epoch'], metadatas, compo_metric, 'Training Accuracy {}'.format(
            # ('(RF)' if args.rf else '(SV)')))
            # plot_training_curves('acc', experts, validation, train['epoch'], metadatas, compo_metric,
            #                      'Validation Accuracy {}'.format(
            # ('(RF)' if args.rf else '(SV)')))
            # plot_training_curves('entropy', experts, validation, train['epoch'], metadatas, compo_metric, 'Entropy {}'.format(
            # ('(RF)' if args.rf else '(SV)')))

            skews.append(float(validation['weight_skew', 'mean']))
            skew_errs.append(float(validation['weight_skew', 'std']))
            entropies.append(float(validation['weight_entropy', 'mean']))
            entropy_errs.append(float(validation['weight_entropy', 'std']))
            # ax = axs[i][j]
            # plot_histogram(compo_metric, metric, metadatas, validation, args, experts, ax=ax)
            # plot_histogram(compo_metric, 'acc', metadatas, train, args, experts, save=False)
            # print_validation_compo(validation, experts, metadatas)

        # fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        # fig.subplots_adjust(wspace=.05)

        # ax = axs[0]
        # plot_training_curves('acc', experts, validation, train['epoch'], metadatas, compo_metric, 'Reinforcement', save=False, ax=ax)


        # supervision
        j = len(ents)
        args.rf = 0
        args.ent = 0.0
        print('rf: {}, ent: {}'.format(args.rf, args.ent))
        train, validation, experts, metadatas, last_interaction = load_data(
            args, args.rf, n, args.ent, last_only=last_only, dv=False)
        skews.append(float(validation['weight_skew', 'mean']))
        skew_errs.append(float(validation['weight_skew', 'std']))
        entropies.append(float(validation['weight_entropy', 'mean']))
        entropy_errs.append(float(validation['weight_entropy', 'std']))
        #
        all_skews.append(skews)
        all_skew_errs.append(skew_errs)
        all_entropies.append(entropies)
        all_entropy_errs.append(entropy_errs)

        # plot_training_curves('loss', experts, train, train['epoch'], metadatas, compo_metric, 'Training Loss {}'.format(
        #     ('(RF)' if args.rf else '(SV)')
        # ))
        # plot_training_curves('acc', experts, train, train['epoch'], metadatas, compo_metric, 'Training Accuracy {}'.format(
        #     ('(RF)' if args.rf else '(SV)')))
        # ax = axs[1]
        # plot_training_curves('acc', experts, validation, train['epoch'], metadatas, compo_metric, 'Supervision', ax=ax, save=False)
        # plot_training_curves('entropy', experts, validation, train['epoch'], metadatas, compo_metric, 'Entropy {}'.format(
        #     ('(RF)' if args.rf else '(SV)')))
        # print_validation_compo(validation, experts, metadatas)
        # supervision
        # far right
        # ax = axs[i][j]
        # plot_histogram(compo_metric, metric, metadatas, validation, args, experts, ax=ax)
        # plot_histogram(compo_metric, 'acc', metadatas, train, args, experts, save=False)

    # Plot skews and entropies
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plot_SEs(ents, all_entropies, all_entropy_errs, expert_list, args, title="Expert Weights Entropy", fig=fig, ax=axs[0], left=True)
    plot_SEs(ents, all_skews, all_skew_errs, expert_list, args, title="Expert Weights Skew", fig=fig, ax=axs[1], left=False)
    axs[1].legend(fontsize=18, loc='lower right')

    # axs[1].legend(fontsize=12, loc='lower right')
    # fig.supylabel('Validation Accuracy', fontsize=20)
    # fig.supxlabel('{} Imitation Training Epochs'.format(args.agent.capitalize()), fontsize=20)
    fig.supxlabel(' ', fontsize=22)
    # fig.supxlabel(r'Entropy Regularization Coeff. ($\lambda$)', fontsize=22)
    # fig.supxlabel(LABELS[compo_metric], fontsize=22)
    # fig.supylabel(LABELS[metric], fontsize=22)
    fig.tight_layout()
    fig.savefig(args.savepath)