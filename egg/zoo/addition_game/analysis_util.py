import torch
from pathlib import Path
import pandas as pd
import numpy as np
from typing import TypeVar, Generic, Sequence, Tuple, Iterable


def load_interaction(file):
    x = torch.load(file, map_location=torch.device("cpu"))
    return x


def load_all_interactions(rootdir: str, mode: str = 'train') -> Tuple[Sequence, Sequence]:
    """
    Loads all interactions and epoch list such that they are ordered ascending by random seed.
    """
    p = Path(rootdir)
    random_seeds = [f for f in p.iterdir() if f.is_dir()]
    random_seeds = sorted(random_seeds, key = lambda p: int(str(p).split('_')[-1]))
    dir_for_mode = [str(rs) + '/interactions/{}'.format(mode) for rs in random_seeds]

    all_interactions = []
    global_epoch_list = []
    for rs in dir_for_mode:
        p = Path(rs)
        epochs = [str(f) for f in p.iterdir() if f.is_dir()]
        epochs = sorted(epochs, key=lambda x: int(x.split('_')[-1]))
        all_interactions.append([load_interaction(epoch + '/interaction_gpu0') for epoch in epochs])

        this_rs_epoch_list = [int(str(epoch).split('_')[-1]) for epoch in epochs]
        global_epoch_list = this_rs_epoch_list if \
            not len(global_epoch_list) or len(this_rs_epoch_list) < len(global_epoch_list) else global_epoch_list

    return all_interactions, global_epoch_list


def filter_interactions(
        all_interactions: Iterable[Dict],
        acc_thres: float=0.75
    ):
    filtered_runs = []
    for i, rs in enumerate(all_interactions):
        if np.max([torch.mean(epoch.aux['acc']) for epoch in rs]) > acc_thres:
            filtered_runs.append(i)

    filtered_interactions = [all_interactions[rs] for rs in filtered_runs]
    return filtered_interactions, filtered_runs


def filter_interactions_on_runs(all_interactions: Iterable[Dict], runs: Sequence[int]):
    return [all_interactions[rs] for rs in runs]


def get_last_val_messages(filtered_interactions: list) -> Iterable[pd.DataFrame]:
    last_val_interactions = [interaction_to_df(rs[-1]) for rs in filtered_interactions]
    return last_val_interactions


def get_best_val_messages(filtered_interactions: list) -> Iterable[pd.DataFrame]:
    best_val_epochs = [np.argmax([torch.mean(epoch.aux['acc']) for epoch in rs]) for rs in filtered_interactions]
    best_val_interactions = get_best_messages_based_on_val(filtered_interactions, best_val_epochs)
    return best_val_interactions, best_val_epochs


def get_best_messages_based_on_val(filtered_interactions: list, best_val_epochs: list) -> Iterable[pd.DataFrame]:
    return [interaction_to_df(interaction[best_val_epochs[i]]) for i, interaction in enumerate(filtered_interactions)]


def interaction_to_df(interaction: Dict) -> pd.DataFrame:
    df_interaction = {}
    n_inputs = int(interaction.sender_input.shape[1] / 2)

    df_interaction['x'] = np.argmax(np.array(interaction.sender_input)[:, :n_inputs], axis=1)
    df_interaction['y'] = np.argmax(np.array(interaction.sender_input)[:, n_inputs:], axis=1)
    df_interaction['sum'] = np.array(interaction.labels)
    df_interaction['symbol'] = np.array(interaction.message)[:, :1].flatten()
    df_interaction['acc'] = np.array(interaction.aux['acc'])
    df_interaction['output'] = np.argmax(np.array(interaction.receiver_output), axis=1)
    df_interaction = pd.DataFrame(df_interaction)

    return df_interaction