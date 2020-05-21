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
def isin(needle, haystack):
    for i in range(len(haystack)):
        if needle == haystack[i]:
            return True
    return False


@nb.njit
def _fit(interactions, user_items, item_idx, regularization, learning_rate, epochs, verbose, x_uf, x_if, w_i, w_if, v_u, v_i, v_uf, v_if):
    """private JIT model-fitting function

    :param interactions: np.array[int32] of observed [user_idx, item_idx] iteractions
    :param user_items: typed dict [int32 -> int32[:]] mapping user_idx to set of observed item_idx
    :param item_idx: np.array[int32] of unique item_idx values found in interactions data
    :param regularization: L2 regularization penalty
    :param learning_rate: SGD weight update learning rate
    :param epochs: number of training epochs (full passes through observed interactions)
    :param verbose: whether to print training epoch and penalized log-likelihood to console
    :return: updated model weights (w_i, w_if, v_u, v_i, v_uf, v_if)
    """

    P = x_uf.shape[1]
    Q = x_if.shape[1]
    F = v_i.shape[1]

    for epoch in range(epochs):

        shuffle_index = np.arange(len(interactions))
        np.random.shuffle(shuffle_index)

        interactions = interactions[shuffle_index]
        log_likelihood = 0.0

        for row in range(len(interactions)):

            u = interactions[row, 0]
            i = interactions[row, 1]
            n_items = len(item_idx)

            while True:
                j = int(n_items * random.random())
                if not isin(j, user_items[u]):
                    break

            pu_i = w_i[i] - w_i[j]
            pu_if = np.dot(x_if[i] - x_if[j], w_if)
            pu_ui = np.dot(v_i[i] - v_i[j], v_u[u])
            pu_u_if = np.dot(x_if[i] - x_if[j], np.dot(v_if, v_u[u]))
            pu_i_uf = np.dot(x_uf[u], np.dot(v_uf, v_i[i] - v_i[j]))
            pu_uf_if = np.dot(np.dot(v_uf.T, x_uf[u]), np.dot(v_if.T, x_if[i] - x_if[j]))

            pairwise_utility = pu_i + pu_if + pu_ui + pu_u_if + pu_i_uf + pu_uf_if
            log_likelihood += np.log(1 / (1 + np.exp(-pairwise_utility)))

            d_con = 1.0 / (np.exp(pairwise_utility) + 1.0)
            d_reg = 2.0 * regularization

            d_w_i = 1.0
            d_w_j = -1.0
            d_w_if = x_if[i] - x_if[j]

            d_v_u = v_i[i] - v_i[j] + np.dot(v_if.T, x_if[i] - x_if[j])
            d_v_i = v_u[u] + np.dot(v_uf.T, x_uf[u])
            d_v_j = -v_u[u] - np.dot(v_uf.T, x_uf[u])

            d_v_uf = np.empty((P, F), np.float32)
            d_v_if = np.empty((Q, F), np.float32)

            for f in range(F):
                for p in range(P):
                    if (x_uf[u][p]) == 0.0:
                        d_v_uf[p, f] = 0.0
                    else:
                        d_v_uf[p, f] = (x_uf[u][p]) * (v_i[i][f] - v_i[j][f] + np.dot(v_if.T[f], x_if[i] - x_if[j]))
                for q in range(Q):
                    if (x_if[i][q] - x_if[j][q]) == 0.0:
                        d_v_if[q, f] = 0.0
                    else:
                        d_v_if[q, f] = (x_if[i][q] - x_if[j][q]) * (v_u[u][f] + np.dot(v_uf.T[f], x_uf[u]))

            w_i[i] += learning_rate * ((d_con * d_w_i)  - (d_reg * w_i[i]))
            w_i[j] += learning_rate * ((d_con * d_w_j)  - (d_reg * w_i[j]))
            w_if   += learning_rate * ((d_con * d_w_if) - (d_reg * w_if))
            v_u[u] += learning_rate * ((d_con * d_v_u)  - (d_reg * v_u[u]))
            v_i[i] += learning_rate * ((d_con * d_v_i)  - (d_reg * v_i[i]))
            v_i[j] += learning_rate * ((d_con * d_v_j)  - (d_reg * v_i[j]))
            v_uf   += learning_rate * ((d_con * d_v_uf) - (d_reg * v_uf))
            v_if   += learning_rate * ((d_con * d_v_if) - (d_reg * v_if))

        penalty = 0.0
        penalty += np.sum(regularization * np.square(w_i))
        penalty += np.sum(regularization * np.square(w_if))
        penalty += np.sum(regularization * np.square(v_u))
        penalty += np.sum(regularization * np.square(v_i))
        penalty += np.sum(regularization * np.square(v_uf))
        penalty += np.sum(regularization * np.square(v_if))

        if verbose:
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

    n_scores = len(pairs)
    scores = np.empty(n_scores, dtype=np.float32)

    for row in range(n_scores):

        u = pairs[row, 0]
        i = pairs[row, 1]

        if np.isnan(u) or np.isnan(i):
            scores[row] = np.nan
        else:

            u = int(u)
            i = int(i)

            item = w_i[i]
            item_features = np.dot(x_if[i], w_if)
            item_user = np.dot(v_i[i], v_u[u])
            user_item_features = np.dot(x_if[i], np.dot(v_if, v_u[u]))
            item_user_features = np.dot(x_uf[u], np.dot(v_uf, v_i[i]))
            feature_interactions = np.dot(np.dot(v_uf.T, x_uf[u]), np.dot(v_if.T, x_if[i]))

            item_utility = item + item_features + item_user + user_item_features + item_user_features + feature_interactions
            scores[row] = item_utility

    return scores


@nb.njit
def _recommend_for_users(users, user_items, n_items, filter_previous, x_uf, x_if, w_i, w_if, v_u, v_i, v_uf, v_if):
    """internal JIT user scoring function

    :param users: np.array[float32] of user indexes potentially containing np.nan
    :param user_items: typed dict [int32 -> int32[:]] mapping user_idx to set of observed item_idx
    :param n_items: number of recommended items to generate for each user
    :param filter_previous: whether to exclude previously observed items
    :return: np.array[float32] where rows are users and columns are recommended items
    """

    n_users = len(users)
    rec_items = np.empty((n_users, n_items), dtype=np.float32)

    for row in range(n_users):

        u = users[row]
        if np.isnan(u):
            rec_items[row] = np.full(n_items, np.nan, dtype=np.float32)
        else:

            u = int(u)

            item = w_i
            item_features = np.dot(x_if, w_if)
            item_user = np.dot(v_i, v_u[u])
            user_item_features = np.dot(x_if, np.dot(v_if, v_u[u]))
            item_user_features = np.dot(np.dot(v_uf, v_i.T).T, x_uf[u])
            feature_interactions = np.dot(np.dot(x_if, v_if), np.dot(v_uf.T, x_uf[u]))

            item_utilities = item + item_features + item_user + user_item_features + item_user_features + feature_interactions
            ranked_items = np.argsort(item_utilities)[::-1]
            selected_items = np.empty(n_items, dtype=np.int32)

            s = 0
            for item in ranked_items:
                if filter_previous and isin(item, user_items[u]):
                    continue
                else:
                    selected_items[s] = item
                    s += 1
                if s == n_items:
                    break

            rec_items[row] = selected_items.astype(np.float32)

    return rec_items

