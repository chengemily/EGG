import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import random
import io
import matplotlib.pyplot as plt


def make_image(ax, map, traj, players, attention_weights=None):
    """

    :param test_map: (Tensor) map features
    :param traj: (Tensor) trajectories (player, features, timesteps). len(features) == 5: pos, vel, angle
    :return: image of dimension (height, width, channels)
    """
    # Plot map on matplotlib
    plot_map(ax, map)

    # Plot trajectories on matplotlib using attention as alpha weights
    for player in players:
        draw_path(ax, traj, player, attention_weights=attention_weights)

    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-0.3, 1.3])

def plot_map(ax, map):
    map = map[:, 1:] # don't need object id

    # features: x, y, r, b, g, big, small, square, triangle
    big_size = 200
    small_size = 100

    # filter out squares
    shapes = {}
    shapes['s'] = [obj for obj in map if obj[7] == 1]

    # filter out triangles
    shapes['^'] = [obj for obj in map if obj[8] == 1]

    for shape_category in shapes:
        xs, ys = [shape[0] for shape in shapes[shape_category]], [shape[1] for shape in shapes[shape_category]]
        colors = ['red' if shape[2] == 1 else ('blue' if shape[3] == 1 else 'green') for shape in shapes[shape_category]]
        sizes = [big_size * shape[5] + small_size * shape[6] for shape in shapes[shape_category]]
        ax.scatter(xs, ys, s=sizes, c=colors, marker=shape_category)

def draw_path(ax, traj, player, num_vectors=8, attention_weights=None, startlabel='Start'):
    """

    :param with_path_gradient: (bool) whether to do a gradient on the path drawing
    :param map: (plt ax)
    :param traj:
    :param angles: (list(float))
    :param player: 0, 1, or (0, 1)
    :param num_vectors
    :return:
    """
    xs, ys = traj[:, player, 0].squeeze(0), traj[:, player, 1].squeeze(0)
    xvels, yvels = traj[:, player, 2].squeeze(0), traj[:, player, 3].squeeze(0)
    angles = traj[:, player, 4].squeeze(0)
    color = 'r' if player else 'b'

    # path itself
    if attention_weights is None:
        ax.plot(xs, ys, color=color)
    else:
        # Plot the path with a gradient
        for t in range(len(xs) - 1):
            ax.plot(xs[t:t + 2], ys[t:t + 2], alpha=attention_weights[t].item(), color=color)

    # cross at start point
    if startlabel:
        ax.plot(xs[0], ys[0], '+k', label=startlabel, markersize=3)

    # make vectors, color by sign of speed
    step = len(xs) // num_vectors
    X, Y = [x for x in xs[step::step]], [y for y in ys[step::step]]
    U = [7 * xvel for xvel in xvels[step::step]]
    V = [7 * yvel for yvel in yvels[step::step]]
    A = angles[step::step]

    for i in range(len(U)):
        # If the velocity is 0, edit it with the angle and set an arbitrary short length
        if U[i] == 0 or V[i] == 0:
            U[i], V[i] = 0.08 * np.cos(A[i]), 0.08 * np.sin(A[i])

    ax.quiver(X, Y, U, V, color=color)


