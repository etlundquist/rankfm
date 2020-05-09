"""
rankfm model tuning and evaluation functions
"""

import os
import sys

import numpy as np
import pandas as pd


def precision_at_k(model, test_interactions, k=10):
    """evaluate precision wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :return: overall average precision (scalar)
    """

    test_users = np.unique(test_interactions.iloc[:, 0])
    train_users = model.user_idx.map(model.index_to_user)
    common_users = list(set(test_users) & set(train_users))

    test_user_items = pd.DataFrame(test_interactions.copy(), columns=['user_id', 'item_id'])
    common_user_items = test_user_items[test_user_items.user_id.isin(common_users)].groupby('user_id')['item_id'].apply(set).to_dict()

    recommendations = model.recommend_for_users(common_users)
    precision = np.mean([len(set(recommendations.loc[key]) & set(val)) / len(set(recommendations.loc[key])) for key, val in common_user_items.items()])
    return precision


def recall_at_k(model, test_interactions, k=10):
    """evaluate recall wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :return: overall average recall (scalar)
    """

    test_users = np.unique(test_interactions.iloc[:, 0])
    train_users = model.user_idx.map(model.index_to_user)
    common_users = list(set(test_users) & set(train_users))

    test_user_items = pd.DataFrame(test_interactions.copy(), columns=['user_id', 'item_id'])
    common_user_items = test_user_items[test_user_items.user_id.isin(common_users)].groupby('user_id')['item_id'].apply(set).to_dict()

    recommendations = model.recommend_for_users(common_users)
    recall = np.mean([len(set(recommendations.loc[key]) & set(val)) / len(set(val)) for key, val in common_user_items.items()])
    return recall