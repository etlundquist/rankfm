"""
rankfm private JIT numba functions used for training/scoring/recommendation
- each numba JIT function will be called by a corresponding python interface function
- all numba functions operate on iternal user_idx and item_idx values
- raw user/item identifiers are mapped to index positions prior to calling JIT functions
"""

import random
import numpy as np
import numba as nb


@nb.njit
def isin_1(item, items):
    """linear search for a given item in a sorted array"""

    for i in items:
        if item == i:
            return True
    return False


@nb.njit
def isin_2(item, items):
    """binary search for a given item in a sorted array"""

    remaining = items.copy()
    lo = 0
    hi = len(remaining)
    md = int((lo + hi) / 2)

    while True:

        if remaining[md] == item:
            return True
        elif (lo == md) or (hi == md):
            return False
        elif remaining[md] < item:
            remaining = remaining[md:]
        else:
            remaining = remaining[:md]

        lo = 0
        hi = len(remaining)
        md = int((lo + hi) / 2)


@nb.njit
def assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if):
    """assert all model weights are finite"""

    assert np.isfinite(np.sum(w_i)), "item weights [w_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(w_if)), "item feature weights [w_if] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_u)), "user factors [v_u] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_i)), "item factors [v_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_uf)), "user-feature factors [v_uf] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_if)), "item-feature factors [v_if] are not finite - try decreasing feature/sample_weight magnitudes"


@nb.njit
def reg_penalty(regularization, w_i, w_if, v_u, v_i, v_uf, v_if):
    """calculate the total regularization penalty for all model weights"""

    penalty = 0.0
    penalty += np.sum(regularization * np.square(w_i))
    penalty += np.sum(regularization * np.square(w_if))
    penalty += np.sum(regularization * np.square(v_u))
    penalty += np.sum(regularization * np.square(v_i))
    penalty += np.sum(regularization * np.square(v_uf))
    penalty += np.sum(regularization * np.square(v_if))
    return penalty


@nb.njit
def _fit(interactions, sample_weight, user_items, item_idx, regularization, learning_rate, learning_schedule, learning_exponent, epochs, verbose, x_uf, x_if, w_i, w_if, v_u, v_i, v_uf, v_if):
    """private JIT model-fitting function

    :param interactions: np.array[int32] of observed [user_idx, item_idx] iteractions
    :param sample_weight: vector of importance weights for each observed interaction
    :param user_items: typed dict [int32 -> int32[:]] mapping user_idx to set of observed item_idx
    :param item_idx: np.array[int32] of unique item_idx values found in interactions data
    :param regularization: L2 regularization penalty
    :param learning_rate: SGD weight update learning rate
    :param learning_schedule: tag for adjusting the learning rate as a function of training epoch
    :param learning_exponent: exponent used to calculate the inverse scaling factor
    :param epochs: number of training epochs (full passes through observed interactions)
    :param verbose: whether to print training epoch and penalized log-likelihood to console
    :return: updated model weights (w_i, w_if, v_u, v_i, v_uf, v_if)
    """

    # define matrix dimension shapes
    F = v_i.shape[1]
    I = len(item_idx)

    # create a shuffle index to diversify each training epoch
    n_interaction = len(interactions)
    shuffle_index = np.arange(n_interaction)

    for epoch in range(epochs):

        if learning_schedule == 'constant':
            eta = learning_rate
        elif learning_schedule == 'invscaling':
            eta = learning_rate / (epoch + 1)**learning_exponent
        else:
            raise ValueError('unknown [learning_schedule]')

        np.random.shuffle(shuffle_index)
        interactions = interactions[shuffle_index]
        sample_weight = sample_weight[shuffle_index]
        log_likelihood = 0.0

        for row in range(n_interaction):

            # locate the user (u), observed item (i), and sample weight (sw)
            u = interactions[row, 0]
            i = interactions[row, 1]
            sw = sample_weight[row]

            # randomly sample an unobserved item (j) for the user
            while True:
                j = int(I * random.random())
                if not isin_1(j, user_items[u]):
                    break

            pu_i = w_i[i] - w_i[j]
            pu_if = np.dot(x_if[i] - x_if[j], w_if)
            pu_u_i = np.dot(v_i[i] - v_i[j], v_u[u])
            pu_u_if = np.dot(x_if[i] - x_if[j], np.dot(v_if, v_u[u]))
            pu_i_uf = np.dot(x_uf[u], np.dot(v_uf, v_i[i] - v_i[j]))
            pu_uf_if = np.dot(np.dot(v_uf.T, x_uf[u]), np.dot(v_if.T, x_if[i] - x_if[j]))

            # calculate the pairwise utility score for the (u, i, j) triplet and its associated log-likelihood
            pairwise_utility = pu_i + pu_if + pu_u_i + pu_u_if + pu_i_uf + pu_uf_if
            log_likelihood += np.log(1 / (1 + np.exp(-pairwise_utility)))

            # calculate derivatives of the penalized log-likelihood function wrt to all model weights
            # NOTE: sample weights are applied like frequency weights: gradient updates are scaled up/down as if there were W identical (u, i, j) pairs

            # calculate the outer-derivative [d_LL/d_g(pu)] and regularization derivative [d_LL/d_norm(theta)]
            d_con = 1.0 / (np.exp(pairwise_utility) + 1.0)
            d_reg = 2.0 * regularization

            # calculate the [item] and [user/item factor] derivatives
            d_w_i = 1.0
            d_w_j = -1.0
            d_v_u = v_i[i] - v_i[j] + np.dot(v_if.T, x_if[i] - x_if[j])
            d_v_i = v_u[u] + np.dot(v_uf.T, x_uf[u])
            d_v_j = -v_u[u] - np.dot(v_uf.T, x_uf[u])

            # update the [item] and [user/item factor] weights with a gradient step
            w_i[i] += eta * (sw * (d_con * d_w_i) - (d_reg * w_i[i]))
            w_i[j] += eta * (sw * (d_con * d_w_j) - (d_reg * w_i[j]))
            v_u[u] += eta * (sw * (d_con * d_v_u) - (d_reg * v_u[u]))
            v_i[i] += eta * (sw * (d_con * d_v_i) - (d_reg * v_i[i]))
            v_i[j] += eta * (sw * (d_con * d_v_j) - (d_reg * v_i[j]))

            # get the non-zero indices of user/item features for this (u, i, j) triplet
            x_uf_nz = np.nonzero(x_uf[u])[0]
            x_if_nz = np.nonzero(x_if[i] - x_if[j])[0]

            # update [user-feature-factor] weights for the non-zero user features
            for p in x_uf_nz:
                for f in range(F):
                    d_v_uf = (x_uf[u][p]) * (v_i[i][f] - v_i[j][f] + np.dot(v_if.T[f], x_if[i] - x_if[j]))
                    v_uf[p, f] += eta * (sw * (d_con * d_v_uf) - (d_reg * v_uf[p, f]))

            # update [item-feature] and [item-feature-factor] weights for the non-zero item features
            for q in x_if_nz:
                d_w_if = x_if[i][q] - x_if[j][q]
                w_if[q] += eta * (sw * (d_con * d_w_if) - (d_reg * w_if[q]))
                for f in range(F):
                    d_v_if = (x_if[i][q] - x_if[j][q]) * (v_u[u][f] + np.dot(v_uf.T[f], x_uf[u]))
                    v_if[q, f] += eta * (sw * (d_con * d_v_if) - (d_reg * v_if[q, f]))

        # assert all model weights are finite as of the end of this epoch
        assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if)

        if verbose:
            penalty = reg_penalty(regularization, w_i, w_if, v_u, v_i, v_uf, v_if)
            log_likelihood = round(log_likelihood - penalty, 2)
            print("\ntraining epoch:", epoch)
            print("log likelihood:", log_likelihood)

    return w_i, w_if, v_u, v_i, v_uf, v_if


@nb.njit
def _predict(pairs, x_uf, x_if, w_i, w_if, v_u, v_i, v_uf, v_if):
    """private JIT user/item pair scoring function

    :param pairs: np.array[int32] of [user_idx, item_idx] pairs to score
    :return: np.array[float32] of corresponding utility scores
    """

    # initialize the scores vector
    n_scores = len(pairs)
    scores = np.empty(n_scores, dtype=np.float32)

    for row in range(n_scores):

        # locate the user/item to score
        u = pairs[row, 0]
        i = pairs[row, 1]

        # set the score to nan if the user or item not found
        if np.isnan(u) or np.isnan(i):
            scores[row] = np.nan
        else:

            # calculate the pointwise utility score for the (u, i) pair

            u = int(u)
            i = int(i)

            ut_i = w_i[i]
            ut_if = np.dot(x_if[i], w_if)
            ut_u_i = np.dot(v_i[i], v_u[u])
            ut_u_if = np.dot(x_if[i], np.dot(v_if, v_u[u]))
            ut_i_uf = np.dot(x_uf[u], np.dot(v_uf, v_i[i]))
            ut_uf_if = np.dot(np.dot(v_uf.T, x_uf[u]), np.dot(v_if.T, x_if[i]))

            item_utility = ut_i + ut_if + ut_u_i + ut_u_if + ut_i_uf + ut_uf_if
            scores[row] = item_utility

    return scores


@nb.njit
def _recommend(users, user_items, n_items, filter_previous, x_uf, x_if, w_i, w_if, v_u, v_i, v_uf, v_if):
    """internal JIT user scoring function

    :param users: np.array[float32] of user indexes potentially containing np.nan
    :param user_items: typed dict [int32 -> int32[:]] mapping user_idx to set of observed item_idx
    :param n_items: number of recommended items to generate for each user
    :param filter_previous: whether to exclude previously observed items
    :return: np.array[float32] where rows are users and columns are recommended items
    """

    # initialize the recs matrix
    n_users = len(users)
    rec_items = np.empty((n_users, n_items), dtype=np.float32)

    for row in range(n_users):

        # set the rec item vector to nan if the user not found
        u = users[row]
        if np.isnan(u):
            rec_items[row] = np.full(n_items, np.nan, dtype=np.float32)
        else:
            u = int(u)

            # calculate the pointwise utility scores of all items for the user
            ut_i = w_i
            ut_if = np.dot(x_if, w_if)
            ut_u_i = np.dot(v_i, v_u[u])
            ut_u_if = np.dot(x_if, np.dot(v_if, v_u[u]))
            ut_i_uf = np.dot(np.dot(v_uf, v_i.T).T, x_uf[u])
            ut_uf_if = np.dot(np.dot(x_if, v_if), np.dot(v_uf.T, x_uf[u]))
            item_utilities = ut_i + ut_if + ut_u_i + ut_u_if + ut_i_uf + ut_uf_if

            # get a ranked list of item index positions for the user
            ranked_items = np.argsort(item_utilities)[::-1]
            selected_items = np.empty(n_items, dtype=np.int32)

            # select the topN items for each user, optionally skipping previously observed items
            s = 0
            for item in ranked_items:
                if filter_previous and isin_1(item, user_items[u]):
                    continue
                else:
                    selected_items[s] = item
                    s += 1
                if s == n_items:
                    break

            rec_items[row] = selected_items.astype(np.float32)

    return rec_items

