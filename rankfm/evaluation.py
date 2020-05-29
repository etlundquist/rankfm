"""
rankfm model tuning and evaluation functions
"""

import numpy as np
import pandas as pd
from rankfm.utils import get_data

def hit_rate(model, test_interactions, k=10, filter_previous=False):
    """evaluate hit-rate (any match) wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: the hit rate or proportion of test users with any matching items
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # calculate the hit rate (percentage of users with any relevant recommendation) wrt common users
    hit_rate = np.mean([int(len(set(test_recs.loc[u]) & test_user_items[u]) > 0) for u in comm_user])
    return hit_rate


def reciprocal_rank(model, test_interactions, k=10, filter_previous=False):
    """evaluate reciprocal rank wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: mean reciprocal rank wrt the test users
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # calculate the reciprocal rank (inverse rank of the first relevant recommended item) wrt common users
    match_indexes = [np.where(test_recs.loc[u].isin(set(test_recs.loc[u]) & test_user_items[u]))[0] for u in comm_user]
    reciprocal_rank = np.mean([1 / (np.min(index) + 1) if len(index) > 0 else 0 for index in match_indexes])
    return reciprocal_rank


def discounted_cumulative_gain(model, test_interactions, k=10, filter_previous=False):
    """evaluate discounted cumulative gain wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: mean discounted cumulative gain wrt the test users
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # calculate the discounted cumulative gain (sum of inverse log scaled ranks of relevant items) wrt common users
    match_indexes = [np.where(test_recs.loc[u].isin(set(test_recs.loc[u]) & test_user_items[u]))[0] for u in comm_user]
    discounted_cumulative_gain = np.mean([np.sum(1 / np.log2(index + 2)) if len(index) > 0 else 0 for index in match_indexes])
    return discounted_cumulative_gain


def precision(model, test_interactions, k=10, filter_previous=False):
    """evaluate precision wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: mean precision wrt the test users
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # calculate average precision wrt common users
    precision = np.mean([len(set(test_recs.loc[u]) & test_user_items[u]) / len(test_recs.loc[u]) for u in comm_user])
    return precision


def recall(model, test_interactions, k=10, filter_previous=False):
    """evaluate recall wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: mean recall wrt the test users
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # calculate average recall across wrt common users
    recall = np.mean([len(set(test_recs.loc[u]) & test_user_items[u]) / len(test_user_items[u]) for u in comm_user])
    return recall


def diversity(model, test_interactions, k=10, filter_previous=False):
    """evaluate the diversity of the model recommendations

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: dataframe of cnt/pct of users recommended for each item
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # get the unique set of test users
    test_user_items = pd.DataFrame(get_data(test_interactions), columns=['user_id', 'item_id'])
    test_users = test_user_items['user_id'].unique()

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values

    # stack the recommendations long-format for aggregation
    test_recs = test_recs.stack().reset_index().drop('level_1', axis=1)
    test_recs.columns = ['user_id', 'item_id']

    # calculate the number and percentage of users getting recommended each unique item
    user_counts = test_recs.groupby('item_id')['user_id'].count().to_frame('cnt_users')
    user_counts = user_counts.reindex(model.item_id.values, fill_value=0).sort_values('cnt_users', ascending=False).reset_index()
    user_counts['pct_users'] = user_counts['cnt_users'] / len(comm_user)
    return user_counts


