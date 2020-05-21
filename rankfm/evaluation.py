"""
rankfm model tuning and evaluation functions
"""

import os
import sys

import numpy as np
import pandas as pd


def precision_at_k(model, test_interactions, k=10, filter_previous=False):
    """evaluate precision wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: overall average precision (scalar)
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items mapping
    test_users = np.unique(test_interactions.iloc[:, 0])
    test_user_items = pd.DataFrame(test_interactions.copy(), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()

    # generate topK recommendations for all test users also present in the training data
    top_recs = model.recommend_for_users(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    cmn_user = top_recs.index.values

    # calculate average precision across all common users
    precision = np.mean([len(set(top_recs.loc[user]) & test_user_items[user]) / len(set(top_recs.loc[user])) for user in cmn_user])
    return precision


def recall_at_k(model, test_interactions, k=10, filter_previous=False):
    """evaluate recall wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: overall average recall (scalar)
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items mapping
    test_users = np.unique(test_interactions.iloc[:, 0])
    test_user_items = pd.DataFrame(test_interactions.copy(), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()

    # generate topK recommendations for all test users also present in the training data
    top_recs = model.recommend_for_users(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    cmn_user = top_recs.index.values

    # calculate average recall across all common users
    recall = np.mean([len(set(top_recs.loc[user]) & test_user_items[user]) / len(test_user_items[user]) for user in cmn_user])
    return recall

