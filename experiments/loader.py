import torch
import argparse

import pickle

def load_metadata_from_pkl(filename: str):
    """
    Loads metadata from json, including model checkpoint.
    :param filename: ending in json, metadata object
    :return: json of all objects including checkpointed game
    """
    metadata = pickle.load(open(filename, 'rb'))

    print('Loading checkpoint: ', metadata)
    return metadata