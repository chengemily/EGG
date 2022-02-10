import copy
import argparse
import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compo_vs_generalization.archs import (
    Freezer,
    NonLinearReceiver,
    PlusOneWrapper,
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics
from egg.zoo.compo_vs_generalization.train import *
from experiments.loader import *

def generate_expert_trajectories():
    pass

def test(metadata_path: str):
    checkpoint_wrapper = load_metadata_from_pkl(metadata_path)
    params = checkpoint_wrapper['params']
    print(checkpoint_wrapper['checkpoint_path'])
    params.append('--load_from_checkpoint={}'.format(checkpoint_wrapper['checkpoint_path']))
    main(params, train_mode=False)

if __name__=='__main__':
    test('saved_models/checkpoint_wrapper_randomseed1.pkl')

