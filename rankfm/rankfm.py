"""
rankfm main modeling class
"""

import numpy as np
import pandas as pd

from rankfm._rankfm import _fit, _predict, _recommend
from rankfm.utils import get_data

class RankFM():
    """Factorization Machines for Ranking Problems with Implicit Feedback Data"""

    def __init__(self, factors=10, loss='bpr', max_samples=10, alpha=0.01, beta=0.1, sigma=0.1, learning_rate=0.1, learning_schedule='constant', learning_exponent=0.25):
        """store hyperparameters and initialize internal data

        :param factors: latent factor rank
        :param loss: optimization/loss function to use for training: ['bpr', 'warp']
        :param max_samples: maximum number of negative samples to draw for WARP loss
        :param alpha: L2 regularization penalty on [user, item] model weights
        :param beta: L2 regularization penalty on [user-feature, item-feature] model weights
        :param sigma: standard deviation to use for random initialization of factor weights
        :param learning_rate: initial learning rate for gradient step updates
        :param learning_schedule: schedule for adjusting learning rates by training epoch: ['constant', 'invscaling']
        :param learning_exponent: exponent applied to epoch number to adjust learning rate: scaling = 1 / pow(epoch + 1, learning_exponent)
        :return: None
        """

        # validate user inputs
        assert isinstance(factors, int) and factors >= 1, "[factors] must be a positive integer"
        assert isinstance(loss, str) and loss in ('bpr', 'warp'), "[loss] must be in ('bpr', 'warp')"
        assert isinstance(max_samples, int) and max_samples > 0, "[max_samples] must be a positive integer"
        assert isinstance(alpha, float) and alpha > 0.0, "[alpha] must be a positive float"
        assert isinstance(beta, float) and beta > 0.0, "[beta] must be a positive float"
        assert isinstance(sigma, float) and sigma > 0.0, "[sigma] must be a positive float"
        assert isinstance(learning_rate, float) and learning_rate > 0.0, "[learning_rate] must be a positive float"
        assert isinstance(learning_schedule, str) and learning_schedule in ('constant', 'invscaling'), "[learning_schedule] must be in ('constant', 'invscaling')"
        assert isinstance(learning_exponent, float) and learning_exponent > 0.0, "[learning_exponent] must be a positive float"

        # store hyperparameters
        self.factors = factors
        self.loss = loss
        self.max_samples = max_samples
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.learning_exponent = learning_exponent

        # set/clear initial model state
        self._reset_state()


    # --------------------------------
    # begin private method definitions
    # --------------------------------


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

        # user/item interactions and sample importance weights
        self.interactions = None
        self.sample_weight = None

        # dictionary of observed items for each user
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


    def _init_all(self, interactions, user_features=None, item_features=None, sample_weight=None):
        """index the raw interaction and user/item features data to numpy arrays

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        # check user data inputs
        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        # save the unique lists of users/items in terms of original identifiers
        interactions_df = pd.DataFrame(get_data(interactions), columns=['user_id', 'item_id'])
        self.user_id = pd.Series(np.sort(np.unique(interactions_df['user_id'])))
        self.item_id = pd.Series(np.sort(np.unique(interactions_df['item_id'])))

        # create zero-based index position to identifier mappings
        self.index_to_user = self.user_id
        self.index_to_item = self.item_id

        # create reverse mappings from identifiers to zero-based index positions
        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.item_to_index = pd.Series(data=self.index_to_item.index, index=self.index_to_item.values)

        # store unique values of user/item indexes and observed interactions for each user
        self.user_idx = np.arange(len(self.user_id), dtype=np.int32)
        self.item_idx = np.arange(len(self.item_id), dtype=np.int32)

        # map the interactions to internal index positions
        self._init_interactions(interactions, sample_weight)

        # map the user/item features to internal index positions
        self._init_features(user_features, item_features)

        # initialize the model weights after the user/item/feature dimensions have been established
        self._init_weights(user_features, item_features)


    def _init_interactions(self, interactions, sample_weight):
        """map new interaction data to existing internal user/item indexes

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        # check user data inputs
        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        # map the raw user/item identifiers to internal zero-based index positions
        # NOTE: any user/item pairs not found in the existing indexes will be dropped

        self.interactions = pd.DataFrame(get_data(interactions).copy(), columns=['user_id', 'item_id'])
        self.interactions['user_id'] = self.interactions['user_id'].map(self.user_to_index).astype(np.int32)
        self.interactions['item_id'] = self.interactions['item_id'].map(self.item_to_index).astype(np.int32)
        self.interactions = self.interactions.rename({'user_id': 'user_idx', 'item_id': 'item_idx'}, axis=1).dropna()

        # store the sample weights internally or create a vector of ones if not passed
        if sample_weight is not None:
            assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
            assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
            assert len(sample_weight) == len(interactions), "[sample_weight] must have the same length as [interactions]"
            self.sample_weight = np.ascontiguousarray(get_data(sample_weight), dtype=np.float32)
        else:
            self.sample_weight = np.ones(len(self.interactions), dtype=np.float32)

        # create python/numba lookup dictionaries containing the set of observed items for each user
        # NOTE: the typed numba dictionary will be used to sample unobserved items during training
        # NOTE: the interactions data must be converted to np.ndarray prior to training to use @njit
        # NOTE: if the model has been previously fit extend rather than replace the itemset for each user

        if self.is_fit:
            new_user_items = self.interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
            self.user_items = {user: np.sort(np.array(list(set(self.user_items[user]) | set(new_user_items[user])), dtype=np.int32)) for user in self.user_items.keys()}
        else:
            self.user_items = self.interactions.sort_values(['user_idx', 'item_idx']).groupby('user_idx')['item_idx'].apply(np.array, dtype=np.int32).to_dict()

        # format the interactions data as a c-contiguous integer array for cython
        self.interactions = np.ascontiguousarray(self.interactions, dtype=np.int32)



    def _init_features(self, user_features=None, item_features=None):
        """initialize the user/item features given existing internal user/item indexes

        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :return: None
        """

        # store the user features as a ndarray [UxP] row-ordered by user index position
        if user_features is not None:
            x_uf = pd.DataFrame(user_features.copy())
            x_uf = x_uf.set_index(x_uf.columns[0])
            x_uf.index = x_uf.index.map(self.user_to_index)
            if np.array_equal(sorted(x_uf.index.values), self.user_idx):
                self.x_uf = np.ascontiguousarray(x_uf.sort_index(), dtype=np.float32)
            else:
                raise KeyError('the users in [user_features] do not match the users in [interactions]')
        else:
            self.x_uf = np.zeros([len(self.user_idx), 1], dtype=np.float32)

        # store the item features as a ndarray [IxQ] row-ordered by item index position
        if item_features is not None:
            x_if = pd.DataFrame(item_features.copy())
            x_if = x_if.set_index(x_if.columns[0])
            x_if.index = x_if.index.map(self.item_to_index)
            if np.array_equal(sorted(x_if.index.values), self.item_idx):
                self.x_if = np.ascontiguousarray(x_if.sort_index(), dtype=np.float32)
            else:
                raise KeyError('the items in [item_features] do not match the items in [interactions]')
        else:
            self.x_if = np.zeros([len(self.item_idx), 1], dtype=np.float32)


    def _init_weights(self, user_features, item_features):
        """initialize model weights given user/item and user_feature/item_feature indexes/shapes"""

        # initialize scalar weights as ndarrays of zeros
        self.w_i = np.zeros(len(self.item_idx)).astype(np.float32)
        self.w_if = np.zeros(self.x_if.shape[1]).astype(np.float32)

        # initialize latent factors by drawing random samples from a normal distribution
        self.v_u = np.random.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors)).astype(np.float32)
        self.v_i = np.random.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors)).astype(np.float32)

        # randomly initialize user feature factors if user features were supplied
        # NOTE: set all user feature factor weights to zero to prevent random scoring influence otherwise
        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = np.random.normal(loc=0, scale=scale, size=[self.x_uf.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_uf = np.zeros([self.x_uf.shape[1], self.factors], dtype=np.float32)

        # randomly initialize item feature factors if item features were supplied
        # NOTE: set all item feature factor weights to zero to prevent random scoring influence otherwise
        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = np.random.normal(loc=0, scale=scale, size=[self.x_if.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_if = np.zeros([self.x_if.shape[1], self.factors], dtype=np.float32)


    # -------------------------------
    # begin public method definitions
    # -------------------------------


    def fit(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """clear previous model state and learn new model weights using the input data

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        self._reset_state()
        self.fit_partial(interactions, user_features, item_features, sample_weight, epochs, verbose)


    def fit_partial(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """learn or update model weights using the input data and resuming from the current model state

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        if self.is_fit:
            self._init_interactions(interactions, sample_weight)
            self._init_features(user_features, item_features)
        else:
            self._init_all(interactions, user_features, item_features, sample_weight)

        # determine the number of negative samples to draw depending on the loss function
        # NOTE: if [loss == 'bpr'] -> [max_samples == 1] and [multiplier ~= 1] for all updates
        # NOTE: the [multiplier] is scaled by total number of items so it's always [0, 1]

        if self.loss == 'bpr':
            max_samples = 1
        elif self.loss == 'warp':
            max_samples = self.max_samples
        else:
            raise ValueError('[loss] function not recognized')

        # NOTE: the cython internal fit method updates the model weights in place via memoryviews
        _fit(
            self.interactions,
            self.sample_weight,
            self.user_items,
            self.x_uf,
            self.x_if,
            self.w_i,
            self.w_if,
            self.v_u,
            self.v_i,
            self.v_uf,
            self.v_if,
            self.alpha,
            self.beta,
            self.learning_rate,
            self.learning_schedule,
            self.learning_exponent,
            max_samples,
            epochs,
            verbose
        )

        self.is_fit = True
        return self


    def predict(self, pairs, cold_start='nan'):
        """calculate the predicted pointwise utilities for all (user, item) pairs

        :param pairs: dataframe of [user, item] pairs to score
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') user/item pairs not found in training data
        :return: np.array of real-valued model scores
        """

        # check user data inputs
        assert isinstance(pairs, (np.ndarray, pd.DataFrame)), "[pairs] must be np.ndarray or pd.dataframe"
        assert pairs.shape[1] == 2, "[pairs] should be: [user_id, item_id]"
        assert self.is_fit, "you must fit the model prior to generating predictions"

        pred_pairs = pd.DataFrame(get_data(pairs).copy(), columns=['user_id', 'item_id'])
        pred_pairs['user_id'] = pred_pairs['user_id'].map(self.user_to_index)
        pred_pairs['item_id'] = pred_pairs['item_id'].map(self.item_to_index)
        pred_pairs = np.ascontiguousarray(pred_pairs, dtype=np.float32)

        scores = _predict(
            pred_pairs,
            self.x_uf,
            self.x_if,
            self.w_i,
            self.w_if,
            self.v_u,
            self.v_i,
            self.v_uf,
            self.v_if
        )

        if cold_start == 'nan':
            return scores
        elif cold_start == 'drop':
            return scores[~np.isnan(scores)]
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")


    def recommend(self, users, n_items=10, filter_previous=False, cold_start='nan'):
        """calculate the topN items for each user

        :param users: iterable of user identifiers for which to generate recommendations
        :param n_items: number of recommended items to generate for each user
        :param filter_previous: remove observed training items from generated recommendations
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') users not found in training data
        :return: pandas dataframe where the index values are user identifiers and the columns are recommended items
        """

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to generating recommendations"

        user_idx = np.ascontiguousarray(pd.Series(users).map(self.user_to_index), dtype=np.float32)
        rec_items = _recommend(
            user_idx,
            self.user_items,
            n_items,
            filter_previous,
            self.x_uf,
            self.x_if,
            self.w_i,
            self.w_if,
            self.v_u,
            self.v_i,
            self.v_uf,
            self.v_if
        )
        rec_items = pd.DataFrame(rec_items, index=users).apply(lambda c: c.map(self.index_to_item))

        if cold_start == 'nan':
            return rec_items
        elif cold_start == 'drop':
            return rec_items.dropna(how='any')
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")


    def similar_items(self, item_id, n_items=10):
        """find the most similar items wrt latent factor space representation

        :param item_id: item to search
        :param n_items: number of similar items to return
        :return: np.array of topN most similar items wrt latent factor representations
        """

        # ensure that the model has been fit before attempting to generate predictions
        assert item_id in self.item_id.values, "you must select an [item_id] present in the training data"
        assert self.is_fit, "you must fit the model prior to generating similarities"

        try:
            item_idx = self.item_to_index.loc[item_id]
        except (KeyError, TypeError):
            print("item_id={} not found in training data".format(item_id))

        # calculate item latent representations in F dimensional factor space
        lr_item = self.v_i[item_idx] + np.dot(self.v_if.T, self.x_if[item_idx])
        lr_all_items = self.v_i + np.dot(self.x_if, self.v_if)

        # calculate the most similar N items excluding the search item
        similarities = pd.Series(np.dot(lr_all_items, lr_item)).drop(item_idx).sort_values(ascending=False)[:n_items]
        most_similar = pd.Series(similarities.index).map(self.index_to_item).values
        return most_similar


    def similar_users(self, user_id, n_users=10):
        """find the most similar users wrt latent factor space representation

        :param user_id: user to search
        :param n_users: number of similar users to return
        :return: np.array of topN most similar users wrt latent factor representations
        """

        # ensure that the model has been fit before attempting to generate predictions
        assert user_id in self.user_id.values, "you must select an [user_id] present in the training data"
        assert self.is_fit, "you must fit the model prior to generating similarities"

        try:
            user_idx = self.user_to_index.loc[user_id]
        except (KeyError, TypeError):
            print("user_id={} not found in training data".format(user_id))

        # calculate user latent representations in F dimensional factor space
        lr_user = self.v_u[user_idx] + np.dot(self.v_uf.T, self.x_uf[user_idx])
        lr_all_users = self.v_u + np.dot(self.x_uf, self.v_uf)

        # calculate the most similar N users excluding the search user
        similarities = pd.Series(np.dot(lr_all_users, lr_user)).drop(user_idx).sort_values(ascending=False)[:n_users]
        most_similar = pd.Series(similarities.index).map(self.index_to_user).values
        return most_similar

