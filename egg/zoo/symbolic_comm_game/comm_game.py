import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import gym
import random
import os
import multiprocessing
from collections import OrderedDict
from typing import Sequence
import PIL.Image
import io
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import SubprocVecEnv
from enum import IntEnum, Enum

from modules.plot_image import make_image

class ParallelEnv(SubprocVecEnv):
    def __init__(self, list_of_fns, start_method='spawn'):
        super(ParallelEnv, self).__init__(list_of_fns, start_method=start_method)

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def _flatten_obs(self, obs, space):
        """
        Flatten observations, depending on the observation space.

        :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                    A list or tuple of observations, one per environment.
                    Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
        :return (OrderedDict<list(ndarray>), tuple<list(ndarray)> or list(ndarray)) observations.
                A list of NumPy arrays or an OrderedDict or tuple of list of numpy arrays.
                Each NumPy array has the environment index as its first axis.
        """
        assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
        assert len(obs) > 0, "need observations from at least one environment"

        if isinstance(space, gym.spaces.Dict):
            assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
            assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
            return OrderedDict([(k, [o[k] for o in obs]) for k in space.spaces.keys()])
        elif isinstance(space, gym.spaces.Tuple):
            assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
            obs_len = len(space.spaces)
            return tuple(([o[i] for o in obs] for i in range(obs_len)))
        else:
            return np.stack(obs)


def make_env(env_id, rank, max_teach_iters, max_test_iters, num_test_rounds, seed=0, tasks=None, custom_task=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.init_data(max_teach_iters, max_test_iters, num_test_rounds)
        env.seed(seed + rank)

        if tasks is not None:
            this_task = tasks[rank]
            env.new_task(this_task, custom_task=custom_task)

        return env

    return _init


"""
    The GameModule takes in all actions(movement, utterance, goal prediction)
    of all agents for a given timestep and returns the total cost for that
    timestep.

    Game consists of:
        -num_agents (scalar)
        -num_landmarks (scalar)
        -locations: [num_agents + num_landmarks, 2]
        -physical: [num_agents + num_landmarks, entity_embed_size]
        -utterances: [num_agents, vocab_size]
        -goals: [num_agents, goal_size]
        -location_observations: [num_agents, num_agents + num_landmarks, 2]
        -memories
            -utterance: [num_agents, num_agents, memory_size]
            -physical:[num_agents, num_agents + num_landmarks, memory_size]
            -action: [num_agents, memory_size]

        config needs: -batch_size, -using_utterances, -world_dim, -vocab_size, -memory_size, -num_colors -num_shapes
"""

class Game(nn.Module):

    def __init__(self, config, complete_tasks, random_select=True, custom_task=None):
        """

        :param config:
        :param complete_tasks:
        :param random_select:
        :param custom_task: (str) supports "Touch any object.", "Touch all objects."
        """
        super(GameModule, self).__init__()

        self.complete_tasks = list(complete_tasks.items())
        self.random_select = random_select

        self.batch_size = config.batch_size # scalar: num games in this batch

        if random_select:
            self.env = ParallelEnv([
                make_env('gym_game:comm-game-v0', i, config.max_teach_iters, config.max_test_iters, config.num_test_rounds,
                         custom_task=custom_task) \
                for i in range(self.batch_size)
            ], start_method='spawn')
        else:
            self.env = ParallelEnv([
                make_env('gym_game:comm-game-v0', i,
                         config.max_teach_iters,
                         config.max_test_iters,
                         config.num_test_rounds,
                         tasks=self.complete_tasks,
                         custom_task=custom_task)
                for i in range(self.batch_size)
            ], start_method='spawn')

        self.num_agents = config.num_agents

        self.teacher, self.student = 0, 1

        self.using_cuda = config.use_cuda
        self.use_pretrained_embeddings = config.use_pretrained_embeddings
        self.max_teach_iters = config.max_teach_iters
        self.max_test_iters = config.max_test_iters
        self.num_test_rounds = config.num_test_rounds
        self.num_iters = self.max_teach_iters + self.num_test_rounds * self.max_test_iters

        self.iter = 0

        if self.max_teach_iters == 0:
            self.is_teaching = False
        else:
            self.is_teaching = True

        self.done_with_task = False

        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        self.config = config

        # self.memories = {
        #         "hidden": Variable(torch.zeros(self.batch_size, 2, config.memory_size)),
        #         "encoder_hiddens": Variable(torch.zeros(self.batch_size, 2, self.max_teach_iters, config.memory_size)),
        #         "log_probs": Variable(torch.zeros(self.batch_size, 2, self.num_iters)),
        #         "attention_weights": Variable(torch.zeros(self.batch_size, 1, self.num_test_rounds * self.max_test_iters, self.max_teach_iters)),
        #         'rewards': Variable(torch.zeros(self.batch_size, 2, self.num_iters)),
        #         'entropy': Variable(torch.zeros(self.batch_size, 2, self.num_iters))
        #     }

    """
        Re-initializes memory. This is useful when switching tasks/maps.
        list_of_things_to_reset: list of strings that are the keys of self.memories
    """
    def reset_memory(self, epoch=None, new_test_map=False, new_task=False, custom_task=None):
        config = self.config

        if new_task:
            # give a new task
            self.iter = 0

            if self.random_select:
                new_task = random.choice(self.complete_tasks)
                self.env.env_method('new_task', new_task, custom_task=custom_task)
            else:
                # just reset the env keeping the tasks
                self.env.env_method('reset')

            self.teaching_map = self.env.get_attr('teaching_map')[0]
            self.test_maps = self.env.get_attr('test_maps')[0]
            self.task_str = self.env.get_attr('task_str')[0]

            things_to_reset = set(['hidden', 'encoder_hiddens', 'log_probs', 'attention_weights', 'rewards', 'entropy'])
            # things_to_reset = set()
        elif new_test_map:
            things_to_reset = set(['hidden'])
            # things_to_reset = set()


        # for student in test, hidden is initialized to the last hidden state in teaching.
        if self.using_cuda:
            self.memories = {
                "hidden": Variable(torch.zeros(self.batch_size, 2, config.memory_size).cuda()) \
                    if "hidden" in things_to_reset else self.memories['hidden'],
                "encoder_hiddens": Variable(torch.zeros(self.batch_size, 2, self.max_teach_iters, config.memory_size).cuda()) \
                    if "encoder_hiddens" in things_to_reset else self.memories['encoder_hiddens'],
                "log_probs": Variable(torch.zeros(self.batch_size, 2, num_iters).cuda()) \
                    if "log_probs" in things_to_reset else self.memories['log_probs'],
                "attention_weights": Variable(torch.zeros(self.batch_size, 1, self.num_test_rounds * self.max_test_iters, self.max_teach_iters).cuda()) \
                    if "attention_weights" in things_to_reset else self.memories['attention_weights'],
                'rewards': Variable(torch.zeros(self.batch_size, 2, num_iters).cuda()) \
                    if "rewards" in things_to_reset else self.memories['rewards'],
                'entropy': Variable(torch.zeros(self.batch_size, 2, num_iters).cuda()) \
                    if "entropy" in things_to_reset else self.memories['entropy']
            }
        else:
            self.memories = {
                "hidden": Variable(torch.zeros(self.batch_size, 2, config.memory_size)) \
                    if "hidden" in things_to_reset else self.memories['hidden'],
                "encoder_hiddens": Variable(torch.zeros(self.batch_size, 2, self.max_teach_iters, config.memory_size)) \
                    if "encoder_hiddens" in things_to_reset else self.memories['encoder_hiddens'],
                "log_probs": Variable(torch.zeros(self.batch_size, 2, self.num_iters)) \
                    if "log_probs" in things_to_reset else self.memories['log_probs'],
                "attention_weights": Variable(torch.zeros(self.batch_size, 1, self.num_test_rounds * self.max_test_iters, self.max_teach_iters)) \
                    if "attention_weights" in things_to_reset else self.memories['attention_weights'],
                'rewards': Variable(
                    torch.zeros(self.batch_size, 2, self.num_iters)) \
                    if "rewards" in things_to_reset else self.memories['rewards'],
                'entropy': Variable(
                    torch.zeros(self.batch_size, 2, self.num_iters)) \
                    if "entropy" in things_to_reset else self.memories['entropy']
            }

    """
    Updates game state given all movements and utterances and returns accrued cost
        - movements: [batch_size, num_agents, action]
        - goal_predictions: [batch_size, num_agents, num_agents, config.goal_size]
    Returns:
        - scalar: total reward of all games in the batch
    """
    def forward(self, movements):
        # get the trajectories
        # Just take the first element of the batch
        self.env.step_async(movements)
        obs, reward, done, options = self.env.step_wait()
        self.is_teaching = np.all([info['is_teaching'] for info in options])
        self.done_with_task = np.any(done)

        if self.done_with_task:
            self.avg_score_for_last_task = np.mean([info['points'] for info in options])
            self.avg_F1_for_last_task = np.mean([info['F1'] for info in options])
            self.trajectories = [info['trajectory'] for info in options]
            if self.max_teach_iters > 0: self.is_teaching = True

        next = np.any([info['next_map'] for info in options])
        movement_feat, map_objects = self.process_obs(obs)

        self.iter += 1
        return movement_feat, map_objects, reward, self.done_with_task, next

    def get_avg_score_for_last_task(self):
        # average points across the batch
        return self.avg_score_for_last_task

    def get_avg_F1_for_last_task(self):
        return self.avg_F1_for_last_task

    def get_task_str(self):
        return self.task_str

    def is_teaching_stage(self):
        return self.is_teaching

    def is_done_with_task(self):
        return self.done_with_task

    def process_obs(self, obs):

        def featurize_for_role(player):
            # Define teacher vs student obs
            movement_features = []
            map_features = []
            loc = []

            for single_obs in obs:
                placeholder = torch.zeros(len(single_obs['map'][0]))
                placeholder[:2] = single_obs['movement_features'][:, player, :2]
                rel_map_features = single_obs['map'] - placeholder

                if self.is_teaching_stage():
                    partner = self.student if player == self.teacher else self.teacher

                    movement_features.append(torch.cat([
                                single_obs['movement_features'][:, player, 2:], # vel, a
                                single_obs['movement_features'][:, player, :2] - single_obs['movement_features'][:, partner, :2], # rel position
                                single_obs['movement_features'][:, partner, 2:]
                            ], 1))
                else:
                    movement_features.append(single_obs['movement_features'][:, player, 2:])

                map_features.append(rel_map_features)
                loc.append(torch.cat(
                    [single_obs['movement_features'][:, player, :2],
                     single_obs['movement_features'][:, player, 4].unsqueeze(0)], dim=1)) # x, y, a

            return torch.stack(movement_features), torch.stack(loc), map_features # list

        if self.is_teaching_stage():
            teacher_obs, teacher_loc, teacher_rel_map_features = featurize_for_role(self.teacher)
        else:
            teacher_obs, teacher_loc, teacher_rel_map_features = None, None, None

        student_obs, student_loc, student_rel_map_features = featurize_for_role(self.student)

        return (teacher_obs, student_obs), (teacher_rel_map_features, student_rel_map_features)


    def get_obs(self):
        obs = self.env.env_method('feature')

        if self.config.use_pretrained_embeddings:
            return self.process_obs(obs), torch.stack(self.env.env_method('get_sentence_embed'))
        else:
            return self.process_obs(obs), self.env.env_method('get_fol')

    def generate_images(self, attention_weights, task_str, epoch):
        """

        :param attention_weights: (list(Tensor)) list of attention weights, length=max_teach_iters, using which to alpha
        value the trajectories of the teacher and student in the teaching stage.
        :return:
        """
        # Take first of the batch (it's random anyway)
        trajectory = self.trajectories[0]
        if attention_weights is not None:
            attention_weights = [attn_weight[0,:].squeeze(0).data for attn_weight in attention_weights]

        # Make complete trajectories for test maps
        if self.max_teach_iters > 0:
            teaching_traj = trajectory[:, :, :, :self.max_teach_iters]

        test_trajs = [trajectory
                          [
                            :, :, :,
                            (self.max_teach_iters + test_no * self.max_test_iters):(self.max_teach_iters + (test_no + 1) * self.max_test_iters)
                          ] for test_no in range(3)
                      ]

        # make test trajectories
        if len(attention_weights):
            num_rows_in_plot = int(len(attention_weights) // 3 + 1)
        elif self.max_teach_iters > 0:
            num_rows_in_plot = 2
        else:
            num_rows_in_plot = 1

        # test maps first row, then N visualizations of teaching map
        fig, axs = plt.subplots(num_rows_in_plot, 3, figsize=(15, 5 * num_rows_in_plot))

        if num_rows_in_plot > 1:
            axs[0, 0].set_title('Test Map 1')
            axs[0, 1].set_title('Test Map 2')
            axs[0, 2].set_title('Test Map 3')
            if self.max_teach_iters > 0:
                axs[1, 1].set_title('Teaching Map Trajectories')
        else:
            axs[0].set_title('Test Map 1')
            axs[1].set_title('Test Map 2')
            axs[2].set_title('Test Map 3')

        # Draw test maps
        for i, test_map in enumerate(self.test_maps):
            if self.max_teach_iters > 0:
                make_image(axs[0, i], test_map, test_trajs[i], [self.student])
            else:
                make_image(axs[i], test_map, test_trajs[i], [self.student])

        # Draw teaching maps
        if attention_weights is not None:
            for i, attention_weight in enumerate(attention_weights):
                test_no = i % 3
                make_image(
                    axs[1 + (i // 3), test_no],
                    self.teaching_map, teaching_traj,
                    [self.teacher, self.student],
                    attention_weights=attention_weight)
        elif self.max_teach_iters > 0:
            make_image(
                axs[1, 1],
                self.teaching_map, teaching_traj,
                [self.teacher, self.student])

        fig.suptitle('{} \n t={}'.format(task_str, epoch))

        return fig


