import re

import numpy as np
from collections import defaultdict

import torch

mse = lambda x, y: np.mean((x - y) ** 2)
acc = lambda x, y: np.sum(x == y) / len(x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def binarize(y, thres=3):
    """Given threshold, binarize the ratings.
    """
    return (y >= thres).astype(int)


def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]


def generate_meshgrid(num_user, num_item):
    user = np.arange(num_user)
    item = np.arange(num_item)
    right, left = np.meshgrid(item, user)
    return np.column_stack((left.ravel(), right.ravel()))


def compute_ips(x, y, y_ips=None):
    """
    :param x: features observed(O=1)
    :param y: labels observed(O=1)
    :param y_ips: all the labels (O=0,1)
    :return:
    """
    if y_ips is None:
        one_over_zl = np.ones(len(y))
    else:
        py1 = y_ips.sum() / len(y_ips)
        py0 = 1 - py1
        po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
        py1o1 = y.sum() / len(y)
        py0o1 = 1 - py1o1

        propensity = np.zeros(len(y))

        propensity[y == 0] = (py0o1 * po1) / py0  # p(o=1|y=0)
        propensity[y == 1] = (py1o1 * po1) / py1  # p(o=1|y=1)
        one_over_zl = 1 / propensity

    one_over_zl = torch.Tensor(one_over_zl)
    return one_over_zl

def parse_float_arg(input, prefix):
    p = re.compile(prefix + "_[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    if m is None:
        return None
    input = m.group()
    p = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    return float(m.group())

def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor([0.0], device=x.device)) + torch.log(1+torch.exp(-torch.abs(x)))
