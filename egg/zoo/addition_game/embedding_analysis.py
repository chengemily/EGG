import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import typing
from typing import TypeVar, Generic, Sequence, Iterable, Callable, Dict, Optional
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import TSNE

import egg.core as core
from egg.core import Interaction
from egg.zoo.addition_game.architectures import Receiver, Sender
from egg.zoo.addition_game.data_readers import get_dataloaders
from egg.zoo.addition_game.callbacks import HoldoutEvaluator
from egg.zoo.addition_game.analysis_util import *
from egg.zoo.addition_game.language_analysis import *


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _ = self.model(x)
        return self._features


def load_model(opts: args.Namespace, experiment: str, rs: int) -> Dict[str, FeatureExtractor]:
    checkpoint_path = args.checkpoint_path + experiment + "/rs_{}/final.tar".format(rs)

    # load trainer from file
    receiver = Receiver(n_output=2 * opts.input_size - 1, n_hidden=opts.receiver_hidden)
    sender = Sender(n_hidden=opts.sender_hidden, n_inputs=2 * opts.input_size)

    sender = core.RnnSenderReinforce(
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
    )

    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell,
    )

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        None,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0,
    )
    checkpoint = torch.load(checkpoint_path)
    game.load_state_dict(checkpoint.model_state_dict)

    for name, layer in sender.named_modules():
        layer.__name__ = name
    for name, layer in receiver.named_modules():
        layer.__name__ = name

    sender_features = FeatureExtractor(sender, layers=["hidden_to_output"])
    receiver_features = FeatureExtractor(receiver, layers=['encoder.embedding', 'agent.fc1'])

    return {'sender': sender_features, 'receiver': receiver_features}


def plot_embeds(fig, axes,
                dataset: Iterable,
                messages: Iterable,
                true_sums: Iterable,
                agent_features: Dict[str, FeatureExtractor],
                holdout_cutoff: float=0,
                savepath: Optional[str]=None
                ):
    """
    tSNEs and plots symbol embeddings for the hidden state of Sender and initial and last embeddings of Receiver.
    """
    embeds = {}
    embeds['sender'] = agent_features['sender'](dataset)
    embeds['receiver'] = agent_features['receiver'](messages[:,:-1])
    layer_names = ['hidden state', 'initial embed.', 'second-to-last embed.']

    for i, agent in enumerate(embeds):
        features = embeds[agent]
        for j, layer in enumerate(features):

            # Build TSNE
            tsne_embeddings = features[layer]
            if len(tsne_embeddings.shape) > 2:
                tsne_embeddings = tsne_embeddings.squeeze(1)

            tsne_embeds = TSNE(n_components=2, random_state=0).fit_transform(tsne_embeddings.numpy())

            # Extract whichever portion of the data we're on
            validation, v_labels = tsne_embeds[:holdout_cutoff], true_sums[:holdout_cutoff]

            if holdout_cutoff < len(tsne_embeds):
                holdout, h_labels = tsne_embeds[holdout_cutoff:], true_sums[holdout_cutoff:]

                # Define x, y and make scatterplot
                hx, hy = [embed[0] for embed in holdout], [embed[1] for embed in holdout]
                axes[i + j].scatter(hx, hy, s=40, c=h_labels, marker='+')

                x, y = [embed[0] for embed in validation], [embed[1] for embed in validation]
                im = axes[i + j].scatter(x, y, s=30, c=v_labels)
            else:
                x, y = [embed[0] for embed in validation], [embed[1] for embed in validation]
                im = axes[i + j].scatter(x, y, s=30, c=v_labels)

            axes[i + j].set_title('tSNE symbol embeddings for {}: {}'.format(agent, layer_names[i + j]))

            if i + j == 2:
                fig.colorbar(im, cax=axes[-1])

    if savepath is not None:
        fig.savefig(savepath)


def main(args):
    for experiment in args.experiment:
        interactions, _ = load_all_interactions(args.runs_path + experiment, mode='validation')
        interactions, best_runs = filter_interactions(interactions, acc_thres=0)
        _, best_val_epochs = get_best_val_messages(interactions)
        good_val_acc_interactions = [interaction[best_val_epochs[i]] for i, interaction in enumerate(interactions)]

        if args.holdout:
            holdout_interactions, epochs = load_all_interactions(args.runs_path + experiment, mode='uniform')
            holdout_interactions = filter_interactions_on_runs(holdout_interactions, best_runs)
            good_val_holdout_interactions = [
                interaction[best_val_epochs[i]] for i, interaction in enumerate(holdout_interactions)
            ]

        for i, rs in enumerate(best_runs):
            if not args.holdout:
                savepath = 'images/{}_embeds_{}_run_{}.png'.format(experiment, 'validation', rs)
            else:
                savepath = 'images/{}_embeds_uniform_run_{}.png'.format(experiment, rs)

            agent_features = load_model(args, experiment, rs)

            gridspec = {'width_ratios': [1, 1, 1, 0.1]}
            fig, axes = plt.subplots(1, 4, figsize=(22, 6), gridspec_kw=gridspec)

            # Plot validation representations.
            interaction = good_val_acc_interactions[i]

            if args.holdout:
                holdout_interaction = good_val_holdout_interactions[i]
                tsne_input = torch.cat([interaction.sender_input, holdout_interaction.sender_input])
                tsne_messages = torch.cat([interaction.message, holdout_interaction.message])
                labels = torch.cat([interaction.labels, holdout_interaction.labels])
            else:
                tsne_input = interaction.sender_input
                tsne_messages = interaction.message
                labels = interaction.labels

            plot_embeds(
                fig, axes,
                tsne_input,
                tsne_messages,
                labels,
                agent_features,
                holdout_cutoff=len(interaction.sender_input),
                savepath=savepath
            )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process runs from addition experiment.')
    parser.add_argument('--checkpoint_path',
                        type=str, default='./additions_checkpoints/')
    parser.add_argument('--runs_path',
                        type=str, default='./additions_interactions/')
    parser.add_argument('--experiment',
                        nargs="*",
                        default=['full_6000']
                        )
    parser.add_argument('--holdout', type=bool, default=False)
    parser.add_argument(
        "--input_size",
        type=int,
        default=20,
        help="N, where the range of inputs are 0...N-1",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=6000,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0.25,
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="gru",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="gru",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=200,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=200,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=100,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=100,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )

    args = parser.parse_args()
    main(args)