"""
rankfm main modeling class
"""

import os
import sys

import numpy as np
import pandas as pd

from utils import sample_tuples


class RankFM():
    """Factorization Machines for Ranking Problems with Implicit Feedback Data"""

    def __init__(self, factors=10, learning_rate=0.1, regularization=0.01, sigma=0.1):
        """store hyperparameters and initialize internal data

        :param factors: latent factor rank
        :param learning_rate: learning rate for gradient step weight updates
        :param regularization: L2 regularization penalty on model weights
        :param sigma: standard deviation to use for random initialization of factor weights
        :return: None
        """

        # model hyperparameters
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.sigma = sigma

        # user/item identifiers <--> index positions mappings
        self.index_to_user = None
        self.index_to_item = None
        self.user_to_index = None
        self.item_to_index = None

        # internal raw interactions data and user:items mapping
        self.interactions = None
        self.user_items = None

        # unique lists of all users/items
        self.user_id = None
        self.item_id = None
        self.user_idx = None
        self.item_idx = None

        # number of unique users/items
        self.n_users = None
        self.n_items = None

        # model scalar weights
        self.w_item = None
        self.w_item_features = None

        # model latent factors
        self.v_user = None
        self.v_item = None
        self.v_user_features = None
        self.v_item_features = None

        # internal state info
        self.is_fit = False


    # --------------------------------
    # begin private method definitions
    # --------------------------------


    def _index(self, interactions, user_features=None, item_features=None):
        """index the raw interaction and user/item features data to numpy arrays

        :param interactions: pandas dataframe of observed user/item interactions
        :param user_features: optional pandas dataframe of additional features for each user
        :param item_features: optional pandas dataframe of additional features for each item
        :return: None
        """

        # copy the raw interactions data into a pandas dataframe for internal use
        self.interactions = pd.DataFrame(interactions.copy(), columns=['user_id', 'item_id'])

        # save the unique lists of users/items in terms of original identifiers
        self.user_id = pd.Series(np.sort(np.unique(self.interactions['user_id'])))
        self.item_id = pd.Series(np.sort(np.unique(self.interactions['item_id'])))

        # create zero-based index position to identifier mappings
        self.index_to_user = self.user_id.to_dict()
        self.index_to_item = self.item_id.to_dict()

        # create reverse mappings from identifiers to zero-based index positions
        self.user_to_index = {val: key for key, val in self.index_to_user.items()}
        self.item_to_index = {val: key for key, val in self.index_to_item.items()}

        # convert the internal interactions data from identifiers to index positions
        self.interactions['user_id'] = self.interactions['user_id'].map(self.user_to_index)
        self.interactions['item_id'] = self.interactions['item_id'].map(self.item_to_index)
        self.interactions = self.interactions.rename({'user_id': 'user_idx', 'item_id': 'item_idx'}, axis=1).astype(np.int32)

        # store unique values of user/item indexes and observed interactions for each user
        self.user_idx = np.sort(self.index_to_user.keys())
        self.item_idx = np.sort(self.index_to_item.keys())
        self.user_items = self.interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()

        # store the total cardinality of users/items
        self.n_users = len(self.user_idx)
        self.n_items = len(self.item_idx)

        # store user features internally as a ndarray [UxP] row-ordered by user index position
        if user_features:
            self.user_features = pd.DataFrame(user_features.copy())
            self.user_features = self.user_features.set_index(self.user_features.columns[0]).astype(np.float32)
            self.user_features.index = self.user_features.index.map(self.user_to_index)
            self.user_features = self.user_features.sort_index().to_numpy()
        else:
            self.user_features = np.zeros((self.n_users, 1))

        # store item features internally as a ndarray [IxQ] row-ordered by item index position
        if item_features:
            self.item_features = pd.DataFrame(item_features.copy())
            self.item_features = self.item_features.set_index(self.item_features.columns[0]).astype(np.float32)
            self.item_features.index = self.item_features.index.map(self.item_to_index)
            self.item_features = self.item_features.sort_index().to_numpy()
        else:
            self.item_features = np.zeros((self.n_items, 1))

        # record the number of user/item features
        self.n_user_features = self.user_features.shape[1]
        self.n_item_features = self.item_features.shape[1]

        # initialize the scalar weights as ndarrays of zeros
        self.w_item = np.zeros(self.n_items)
        self.w_item_features = np.zeros(self.n_item_features)

        # initialize the latent factors by drawing random samples from a normal distribution
        self.v_user = np.random.normal(loc=0, scale=self.sigma, size=(self.n_users, self.factors))
        self.v_item = np.random.normal(loc=0, scale=self.sigma, size=(self.n_items, self.factors))
        self.v_user_features = np.random.normal(loc=0, scale=self.sigma, size=(self.n_user_features, self.factors))
        self.v_item_features = np.random.normal(loc=0, scale=self.sigma, size=(self.n_item_features, self.factors))


    def _pointwise_utility(self, u, i):
        """calculate the pointwise utility of a given (user, item) pair

        :param u: user index
        :param i: item index
        :return: scalar utility
        """

        item = self.w_item[i]
        item_features = np.dot(self.item_features[i], self.w_item_features)
        item_user = np.dot(self.v_item[i], self.v_user[u])
        user_item_features = np.dot(self.item_features[i], np.dot(self.v_item_features, self.v_user[u]))
        item_user_features = np.dot(self.user_features[u], np.dot(self.v_user_features, self.v_item[i]))

        utility = item + item_features + item_user + user_item_features + item_user_features
        return utility


    def _pointwise_utilities(self, u):
        """calculate the pointwise utilities of all items for a given user

        :param u: user index
        :return: vector of predicted utilities for all items by index position
        """

        item = self.w_item
        item_features = np.dot(self.item_features, self.w_item_features)
        item_user = np.dot(self.v_item, self.v_user[u])
        user_item_features = np.dot(self.item_features, np.dot(self.v_item_features, self.v_user[u]))
        item_user_features = np.dot(np.dot(self.v_user_features, self.v_item.T).T, self.user_features[u])

        utilities = item + item_features + item_user + user_item_features + item_user_features
        return utilities


    def _pairwise_utility(self, u, i, j):
        """calculate the pairwise utility of an (u, i, j) sample

        :param u: user index
        :param i: observed item index
        :param j: unobserved item index
        :return: pairwise utility score: utility(u, i) - utility(u, j)
        """

        item = self.w_item[i] - self.w_item[j]
        item_features = np.dot(self.item_features[i] - self.item_features[j], self.w_item_features)
        item_user = np.dot(self.v_item[i] - self.v_item[j], self.v_user[u])
        user_item_features = np.dot(self.item_features[i] - self.item_features[j], np.dot(self.v_item_features, self.v_user[u]))
        item_user_features = np.dot(self.user_features[u], np.dot(self.v_user_features, self.v_item[i] - self.v_item[j]))

        utility = item + item_features + item_user + user_item_features + item_user_features
        return utility











    def _gradient_step(self, u, i, j):
        """update current model weights based on the gradient of the log-likelihood function

        :param sample: (u, i, j) index position tuple
        :return: None
        """

        # calculate the pairwise utility of the (u, i, j) pair
        utility = self._pairwise_utility(u, i, j)

        # calculate the outer derivative [d_LL/d_g(theta)] and the regularization derivative [d_pen/d_theta]
        d_con = (1 / (1 + np.exp(utility)))
        d_pen = 2 * self.regularization

        # calculate the inner derivatives [d_g(theta)/d_theta]
        d_w_i = 1
        d_w_j = -1
        d_v_u = self.v_item[i] - self.v_item[j]
        d_v_i = self.v_user[u]
        d_v_j = -self.v_user[u]

        # perform the gradient updates to the relevant model weights
        self.w_item[i] += self.learning_rate * ((d_con * d_w_i) - (d_pen * self.w_item[i]))
        self.w_item[j] += self.learning_rate * ((d_con * d_w_j) - (d_pen * self.w_item[j]))
        self.v_user[u] += self.learning_rate * ((d_con * d_v_u) - (d_pen * self.v_user[u]))
        self.v_item[i] += self.learning_rate * ((d_con * d_v_i) - (d_pen * self.v_item[i]))
        self.v_item[j] += self.learning_rate * ((d_con * d_v_j) - (d_pen * self.v_item[j]))


    def _log_likelihood(self, samples):
        """calculate the regularized log-likelihood of all sample pairs

        :param samples: list of (u, i, j) index position tuples
        :return: current penalized log-likelihood (scalar)
        """

        likelihood = sum([np.log(1 / (1 + np.exp(-self._pairwise_utility(u, i, j)))) for u, i, j in samples])
        penalty = sum([np.sum(self.regularization * np.square(w)) for w in [self.w_item, self.v_user, self.v_item]])
        return likelihood - penalty


    # -------------------------------
    # begin public method definitions
    # -------------------------------


    def fit(self, interactions, user_features=None, item_features=None, epochs=1, verbose=False):
        """train model weights using the interaction data

        :param interactions: pandas dataframe of observed user/item interactions
        :param user_features: pandas dataframe of additional features for each user
        :param item_features: pandas dataframe of additional features for each item
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        # copy the raw interaction data into an internal pandas dataframe
        # and map original user/item identifers to zero-based index positions
        self._index(interactions, user_features, item_features)

        for epoch in range(epochs):

            shuffle_index = np.arange(self.interactions.shape[0])
            np.random.shuffle(shuffle_index)

            shuffle_samples = sample_tuples(self.interactions.iloc[shuffle_index], set(self.item_idx), self.user_items)
            for u, i, j in shuffle_samples:
                self._gradient_step(u, i, j)

            if verbose:
                ll = self._log_likelihood(shuffle_samples)
                print("\ntraining epoch: {}".format(epoch))
                print("penalized log-likelihood: {}".format(round(ll, 2)))

        # set internal model state to fitted and return object reference for chained method calls
        self.is_fit = True
        return self


    def predict(self, interactions, cold_start='nan'):
        """calculate the predicted pointwise utilities for all (user, item) pairs

        :param interactions: pandas dataframe of user/item pairs for which to generate utility scores
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') user/item pairs not found in training data
        :return: vector of real-valued model scores
        """

        # ensure that the model has been fit before attempting to generate predictions
        assert self.is_fit, "you must fit the model prior to generating predictions"

        # map raw user/item identifiers to internal index positions
        pred_interactions = pd.DataFrame(interactions.copy(), columns=['user_id', 'item_id'])
        pred_interactions['user_id'] = pred_interactions['user_id'].map(self.user_to_index)
        pred_interactions['item_id'] = pred_interactions['item_id'].map(self.item_to_index)
        pred_interactions = pred_interactions.rename({'user_id': 'user_idx', 'item_id': 'item_idx'}, axis=1)

        # generate model scores for user/item pairs found in training data and np.nan otherwise
        pairs = zip(pred_interactions['user_idx'], pred_interactions['item_idx'])
        scores = np.array([self._pointwise_utility(int(user), int(item)) if not (np.isnan(user) or np.isnan(item)) else np.nan for user, item in pairs])

        if cold_start == 'nan':
            return scores
        elif cold_start == 'drop':
            return scores[~np.isnan(scores)]
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")



    def recommend_for_users(self, users, n_items=10, filter_previous=False, cold_start='nan'):
        """calculate the topN items for each user

        :param users: list-like of user identifiers for which to generate recommendations
        :param n_items: number of recommended items to generate for each user
        :param filter_previous: remove observed training items from generated recommendations
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') users not found in training data
        :return: pandas dataframe where the index values are user identifiers and the columns are recommended items
        """

        # ensure that the model has been fit before attempting to generate predictions
        assert self.is_fit, "you must fit the model prior to generating recommendations"

        def rec_for_user(user_idx):
            """subroutine to recommend for a single user index"""

            if np.isnan(user_idx):
                return np.full(n_items, np.nan)
            else:
                recs = pd.Series(self._pointwise_utilities(int(user_idx)))
                if filter_previous:
                    recs = recs[~recs.isin(self.user_items[int(user_idx)])]
                return recs.sort_values(ascending=False)[:n_items].index.values

        # generate topN recommended items for all users present in the training data and np.nan otherwise
        user_idx = pd.Series(users).map(self.user_to_index)
        top_n = pd.DataFrame([rec_for_user(user) for user in user_idx], index=users).apply(lambda c: c.map(self.index_to_item))

        if cold_start == 'nan':
            return top_n
        elif cold_start == 'drop':
            return top_n.dropna(how='any')
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")


