import json
import numpy as np
import torch

import egg.core as core
from egg.core.batch import Batch


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def entropy(messages):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except ValueError:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy

def get_grad_norm(agent):
    with torch.no_grad():
        total_norm = 0
        for p in agent.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
    return total_norm

def get_avg_grad_norm(agent):
    with torch.no_grad():
        grad_norms = [p.grad.data.norm(2).item() for p in agent.parameters()]
        print(grad_norms)
        return sum(grad_norms) / len(grad_norms)

def get_max_grad(agent):
    with torch.no_grad():
        return max([math.abs(p.grad.data.item()) for p in agent.parameters()])

def sum_grads(agent):
    with torch.no_grad():
        grad_norms = [p.grad.data.norm(2).item() for p in agent.parameters()]
        print(grad_norms)
        return sum(grad_norms)
