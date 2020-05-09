"""
rankfm utility functions
"""

import os
import sys
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd


def sample_tuple(u, i, items, user_items):
    """sample a single (u, i, j) tuple

    :param u: user_id
    :param i: item_id
    :param items: set of all items
    :param user_items: dictionary where keys are user_id and values are sets of interacted items
    :return: (u, i, j) tuple where item (j) is randomly sampled from non-interacted items
    """

    return u, i, np.random.choice(list(items - user_items[u]))


def sample_tuples(interactions, items, user_items, num_threads=mp.cpu_count()):
    """sample a (u, i, j) tuple for every observed interaction

    :param interactions: pandas dataframe of observed user/item interactions
    :param items: set of all items
    :param user_items: dictionary where keys are user_id and values are sets of interacted items
    :param num_threads: number of parallel computation threads to use
    :return: list of sampled (u, i, j) tuples with one tuple for each observed (u, i) interaction
    """

    pool = mp.Pool(processes=num_threads)
    mp_sample_tuple = functools.partial(sample_tuple, items=items, user_items=user_items)
    samples = pool.starmap(mp_sample_tuple, zip(interactions.iloc[:, 0], interactions.iloc[:, 1]))
    pool.close()
    pool.join()
    return samples

