import argparse
import numpy as np
import torch
import random
import comm_game_config as configs
from modules.model import AgentModule
from modules.random_model import RandomAgentModule
from modules.comm_game import GameModule
from collections import defaultdict
import json
import pdb
from tqdm import tqdm

STUDENT = 1
TEACHER = 0

def discounted_cumsum_right(single_round, gamma):
    """
    Performs a right discounted cumulative sum
    :param single_round: 1 x N
    :param gamma:
    :return:
    """
    transposed_round = torch.transpose(torch.fliplr(single_round), 0, 1)
    returns = torch.zeros_like(transposed_round)

    R = torch.zeros(single_round.shape[0])
    for i, r in enumerate(transposed_round):
        R = r + gamma * R
        returns[i] = R

    return torch.fliplr(torch.transpose(returns, 0, 1))

def discount_rewards(rewards, teach_iters, test_iters, gamma):
    """
    Returns discounted future rewards
    :param rewards:
    :param discount_factor:
    :return:
    """
    returns = torch.zeros_like(rewards)

    if teach_iters > 0:
        teaching_round = rewards[:,:,:teach_iters]

        # it can't handle batches
        for i, single_round in enumerate(teaching_round):
            returns[i,:,:teach_iters] = discounted_cumsum_right(single_round, gamma)

    # 3 test maps
    for test_round in range(3):
        start = teach_iters + (test_round * test_iters)
        end = start + test_iters
        # print('rewards shape', rewards.shape)
        test_round = rewards[:,:,start:end]
        for i, single_round in enumerate(test_round):
            returns[i,:,start:end] = discounted_cumsum_right(single_round, gamma)

    return returns

def get_loss(game, subtract_baseline=True, entropy_term=True, entropy_beta=0.1, discount_factor=1):
    """
    Returns sum of total future loss at each time step.
    :param game:
    :param subtract_baseline: (bool) if True, subtracts the mean divide by (stdev + eps)
    :param entropy_term: (bool) if True, adds a entropy_beta * avg. action distribution entropy over the round.
    :param discount_factor: (0 < float < 1)
    :return: total_loss, total_student_loss, total_teacher_loss
    """
    max_teach_iters = game.max_teach_iters
    max_test_iters = game.max_test_iters

    # Process rewards
    # print('loss:45')
    rewards = game.memories['rewards']
    total_rewards = torch.sum(rewards, dim=[0, 2])
    # print('Rewards')
    # print(rewards[:,STUDENT])
    discounted_future_returns = discount_rewards(rewards, max_teach_iters, max_test_iters, discount_factor)

    # print('Discounted future returns')
    # print(discounted_future_returns[:,STUDENT])
    # print('batched discounted returns shape', discounted_future_returns.shape)
    # print('loss:48')
    if subtract_baseline:
        # calculate mean over all batches
        mean, std = torch.mean(discounted_future_returns, dim=[0,2]), torch.std(discounted_future_returns, dim=2)
        discounted_future_returns = torch.sub(discounted_future_returns, mean.unsqueeze(0).unsqueeze(2))

    # Process log probs, loss
    log_probs = game.memories['log_probs']
    student_log_probs = log_probs[:, STUDENT]
    total_student_loss = torch.sum(torch.mul(-1 * student_log_probs, discounted_future_returns[:,STUDENT]), [0,1])
    # print('TOTAL STUDENT LOSS', total_student_loss)
    total_teacher_loss = torch.tensor(0.0)

    avg_student_entropy = torch.mean(game.memories['entropy'][:, STUDENT])
    avg_teacher_entropy = torch.mean(game.memories['entropy'][:, TEACHER])

    if entropy_term:
        total_student_loss -= avg_student_entropy * entropy_beta

    # print('loss:57')
    if max_teach_iters > 0:
        teacher_log_probs = log_probs[:, TEACHER]
        total_teacher_loss = torch.sum(torch.mul(-1 * teacher_log_probs, discounted_future_returns[:,TEACHER]), [0,1])

        if entropy_term:
            total_teacher_loss -= torch.mean(game.memories['entropy'][:, TEACHER]) * entropy_beta

    total_loss = total_teacher_loss + total_student_loss
    # print('TOTAL LOSS', total_loss)
    # print('loss:60')
    return total_loss, total_teacher_loss, total_student_loss, torch.sum(total_rewards), \
           total_rewards[TEACHER], total_rewards[STUDENT], avg_teacher_entropy, avg_student_entropy

