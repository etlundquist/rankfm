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

        # store hyperparameters
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.sigma = sigma

        # set/clear initial model state
        self._reset_state()


    # -------------------------------------------
    # begin private initializing/indexing methods
    # -------------------------------------------


    def _reset_state(self):
        """initialize/reset all [user/item/feature] indexes and model weights"""

        # user/item ID/INDEX arrays
        self.user_id = None
        self.item_id = None
        self.user_idx = None
        self.item_idx = None

        # user/item ID <--> INDEX mappings
        self.index_to_user = None
        self.index_to_item = None
        self.user_to_index = None
        self.item_to_index = None

        # user/item interactions
        self.interactions = None
        self.user_items = None

        # user/item features
        self.x_uf = None
        self.x_if = None

        # item and item feature weights
        self.w_i = None
        self.w_if = None

        # user/item/user-feature/item-feature latent factors
        self.v_u = None
        self.v_i = None
        self.v_uf = None
        self.v_if = None

        # internal model state
        self.is_fit = False


    def _init_all(self, interactions, user_features=None, item_features=None):
        """index the raw interaction and user/item features data to numpy arrays

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :return: None
        """

        # save the unique lists of users/items in terms of original identifiers
        interactions_df =  pd.DataFrame(interactions, columns=['user_id', 'item_id'])
        self.user_id = pd.Series(np.sort(np.unique(interactions_df['user_id'])))
        self.item_id = pd.Series(np.sort(np.unique(interactions_df['item_id'])))

        # create zero-based index position to identifier mappings
        self.index_to_user = self.user_id
        self.index_to_item = self.item_id

        # create reverse mappings from identifiers to zero-based index positions
        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.item_to_index = pd.Series(data=self.index_to_item.index, index=self.index_to_item.values)

        # store unique values of user/item indexes and observed interactions for each user
        self.user_idx = np.arange(len(self.user_id))
        self.item_idx = np.arange(len(self.item_id))

        # map the interactions to internal index positions
        self._init_interactions(interactions)

        # map the user/item features to internal index positions
        self._init_features(user_features, item_features)

        # initialize the model weights after the user/item/feature dimensions have been established
        self._init_weights(user_features, item_features)


    def _init_interactions(self, interactions):
        """map new interaction data to existing internal user/item indexes

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :return: None
        """

        # NOTE: any user/item ID pairs not found in the existing index will be dropped
        self.interactions = pd.DataFrame(interactions.copy(), columns=['user_id', 'item_id'])
        self.interactions['user_id'] = self.interactions['user_id'].map(self.user_to_index)
        self.interactions['item_id'] = self.interactions['item_id'].map(self.item_to_index)
        self.interactions = self.interactions.rename({'user_id': 'user_idx', 'item_id': 'item_idx'}, axis=1).dropna(how='any').astype(np.int32)
        self.user_items   = self.interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()


    def _init_features(self, user_features=None, item_features=None):
        """initialize the user/item features given existing internal user/item indexes

        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :return: None
        """

        # store the user features as a ndarray [UxP] row-ordered by user index position
        if user_features is not None:
            x_uf = pd.DataFrame(user_features.copy())
            x_uf = x_uf.set_index(x_uf.columns[0]).astype(np.float32)
            x_uf.index = x_uf.index.map(self.user_to_index)
            if np.array_equal(sorted(x_uf.index.values), self.user_idx):
                self.x_uf = x_uf.sort_index().to_numpy()
            else:
                raise KeyError('the users in [user_features] do not match the users in [interactions]')
        else:
            self.x_uf = np.zeros([len(self.user_idx), 1])

        # store the item features as a ndarray [IxQ] row-ordered by item index position
        if item_features is not None:
            x_if = pd.DataFrame(item_features.copy())
            x_if = x_if.set_index(x_if.columns[0]).astype(np.float32)
            x_if.index = x_if.index.map(self.item_to_index)
            if np.array_equal(sorted(x_if.index.values), self.item_idx):
                self.x_if = x_if.sort_index().to_numpy()
            else:
                raise KeyError('the items in [item_features] do not match the items in [interactions]')
        else:
            self.x_if = np.zeros([len(self.item_idx), 1])


    def _init_weights(self, user_features, item_features):
        """initialize model weights given user/item and user_feature/item_feature indexes/shapes"""

        # initialize scalar weights as ndarrays of zeros
        self.w_i = np.zeros(len(self.item_idx))
        self.w_if = np.zeros(self.x_if.shape[1])

        # initialize latent factors by drawing random samples from a normal distribution
        self.v_u = np.random.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors))
        self.v_i = np.random.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors))

        # randomly initialize user feature factors if user features were supplied
        # NOTE: set all user feature factor weights to zero to prevent random scoring influence otherwise
        if user_features is None:
            self.v_uf = np.zeros([self.x_uf.shape[1], self.factors])
        else:
            self.v_uf = np.random.normal(loc=0, scale=self.sigma, size=[self.x_uf.shape[1], self.factors])

        # randomly initialize item feature factors if item features were supplied
        # NOTE: set all item feature factor weights to zero to prevent random scoring influence otherwise
        if item_features is None:
            self.v_if = np.zeros([self.x_if.shape[1], self.factors])
        else:
            self.v_if = np.random.normal(loc=0, scale=self.sigma, size=[self.x_if.shape[1], self.factors])


    # -----------------------------------
    # begin private model-fitting methods
    # -----------------------------------


    def _pointwise_utility(self, u, i):
        """calculate the pointwise utility of a given (user, item) pair

        :param u: user index
        :param i: item index
        :return: scalar predicted utility value
        """

        item = self.w_i[i]
        item_features = np.dot(self.x_if[i], self.w_if)
        item_user = np.dot(self.v_i[i], self.v_u[u])
        user_item_features = np.dot(self.x_if[i], np.dot(self.v_if, self.v_u[u]))
        item_user_features = np.dot(self.x_uf[u], np.dot(self.v_uf, self.v_i[i]))
        feature_interactions = np.dot(np.dot(self.v_uf.T, self.x_uf[u]), np.dot(self.v_if.T, self.x_if[i]))

        utility = item + item_features + item_user + user_item_features + item_user_features + feature_interactions
        return utility


    def _pointwise_utilities(self, u):
        """calculate the pointwise utilities of all items for a given user

        :param u: user index
        :return: vector of predicted utilities for all items
        """

        item = self.w_i
        item_features = np.dot(self.x_if, self.w_if)
        item_user = np.dot(self.v_i, self.v_u[u])
        user_item_features = np.dot(self.x_if, np.dot(self.v_if, self.v_u[u]))
        item_user_features = np.dot(np.dot(self.v_uf, self.v_i.T).T, self.x_uf[u])
        feature_interactions = np.dot(np.dot(self.x_if, self.v_if), np.dot(self.v_uf.T, self.x_uf[u]))

        utilities = item + item_features + item_user + user_item_features + item_user_features + feature_interactions
        return utilities


    def _pairwise_utility(self, u, i, j):
        """calculate the pairwise utility of an (u, i, j) sample

        :param u: user index
        :param i: observed item index
        :param j: unobserved item index
        :return: pairwise utility = pointwise_utility(u, i) - pointwise_utility(u, j)
        """

        item = self.w_i[i] - self.w_i[j]
        item_features = np.dot(self.x_if[i] - self.x_if[j], self.w_if)
        item_user = np.dot(self.v_i[i] - self.v_i[j], self.v_u[u])
        user_item_features = np.dot(self.x_if[i] - self.x_if[j], np.dot(self.v_if, self.v_u[u]))
        item_user_features = np.dot(self.x_uf[u], np.dot(self.v_uf, self.v_i[i] - self.v_i[j]))
        feature_interactions = np.dot(np.dot(self.v_uf.T, self.x_uf[u]), np.dot(self.v_if.T, self.x_if[i] - self.x_if[j]))

        utility = item + item_features + item_user + user_item_features + item_user_features + feature_interactions
        return utility


    def _gradient_step(self, u, i, j):
        """update current model weights based on the gradient of the log-likelihood function

        :param u: user index
        :param i: observed item index
        :param j: unobserved item index
        :return: None
        """

        # calculate the pairwise utility of the (u, i, j) pair
        utility = self._pairwise_utility(u, i, j)

        # calculate the outer derivative [d_LL/d_g(theta)] and the regularization derivative [d_reg/d_theta]
        d_con = 1.0 / (np.exp(utility) + 1.0)
        d_reg = 2.0 * self.regularization

        # [scalar weights (1)] calculate the inner derivatives [d_g(theta)/d_theta]
        d_w_i = 1.0
        d_w_j = -1.0
        d_w_if = self.x_if[i] - self.x_if[j]

        # [user/item (Fx1) factors] calculate the inner derivatives [d_g(theta)/d_theta]
        d_v_u = self.v_i[i] - self.v_i[j] + np.dot(self.v_if.T, self.x_if[i] - self.x_if[j])
        d_v_i = self.v_u[u] + np.dot(self.v_uf.T, self.x_uf[u])
        d_v_j = -self.v_u[u] - np.dot(self.v_uf.T, self.x_uf[u])

        # [user (PxF) and item (QxF) feature factors] calculate the inner derivatives [d_g(theta)/d_theta]
        # NOTE: these nested for loops should be nearly as fast as the matrix multiplication form in the optimized code

        P = len(self.x_uf[u])
        Q = len(self.x_if[i])
        F = len(self.v_i[i])

        d_v_uf = np.empty([P, F])
        d_v_if = np.empty([Q, F])

        for f in range(F):
            for p in range(P):
                if (self.x_uf[u][p]) == 0.0:
                    d_v_uf[p, f] = 0.0
                else:
                    d_v_uf[p, f] = (self.x_uf[u][p]) * (self.v_i[i][f] - self.v_i[j][f] + np.dot(self.v_if.T[f], self.x_if[i] - self.x_if[j]))
            for q in range(Q):
                if (self.x_if[i][q] - self.x_if[j][q]) == 0.0:
                    d_v_if[q, f] = 0.0
                else:
                    d_v_if[q, f] = (self.x_if[i][q] - self.x_if[j][q]) * (self.v_u[u][f] + np.dot(self.v_uf.T[f], self.x_uf[u]))

        # perform the gradient updates to the relevant model weights
        self.w_i[i] += self.learning_rate * ((d_con * d_w_i)  - (d_reg * self.w_i[i]))
        self.w_i[j] += self.learning_rate * ((d_con * d_w_j)  - (d_reg * self.w_i[j]))
        self.w_if   += self.learning_rate * ((d_con * d_w_if) - (d_reg * self.w_if))
        self.v_u[u] += self.learning_rate * ((d_con * d_v_u)  - (d_reg * self.v_u[u]))
        self.v_i[i] += self.learning_rate * ((d_con * d_v_i)  - (d_reg * self.v_i[i]))
        self.v_i[j] += self.learning_rate * ((d_con * d_v_j)  - (d_reg * self.v_i[j]))
        self.v_uf   += self.learning_rate * ((d_con * d_v_uf) - (d_reg * self.v_uf))
        self.v_if   += self.learning_rate * ((d_con * d_v_if) - (d_reg * self.v_if))


    def _log_likelihood(self, samples):
        """calculate the regularized log-likelihood of all sample pairs

        :param samples: list of (u, i, j) index position tuples
        :return: current penalized log-likelihood (scalar)
        """

        likelihood = sum([np.log(1 / (1 + np.exp(-self._pairwise_utility(u, i, j)))) for u, i, j in samples])
        penalty = sum([np.sum(self.regularization * np.square(w)) for w in [self.w_i, self.w_if, self.v_u, self.v_i, self.v_uf, self.v_if]])
        return likelihood - penalty


    # --------------------
    # begin public methods
    # --------------------


    def fit(self, interactions, user_features=None, item_features=None, epochs=1, verbose=False):
        """clear previous model state and learn new model weights using the input data

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        self._reset_state()
        self.fit_partial(interactions, user_features, item_features, epochs, verbose)


    def fit_partial(self, interactions, user_features=None, item_features=None, epochs=1, verbose=False):
        """learn or update model weights using the input data and resuming from the current model state

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        if self.is_fit:
            self._init_interactions(interactions)
            self._init_features(user_features, item_features)
        else:
            self._init_all(interactions, user_features, item_features)

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

        self.is_fit = True
        return self


    def predict(self, interactions, cold_start='nan'):
        """calculate the predicted pointwise utilities for all (user, item) pairs

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
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


