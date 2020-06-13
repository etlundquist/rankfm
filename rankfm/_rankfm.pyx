#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# -----------------------
# [C/Python] dependencies
# -----------------------

from libc.stdlib cimport malloc, free
from libc.math cimport log, exp, pow

cimport cython
cimport rankfm.mt19937ar as mt

import numpy as np

# --------------------
# [C] helper functions
# --------------------

cdef int lsearch(int item, int *items, int n) nogil:
    """linear search for a given item in a sorted array of items"""

    cdef int i
    for i in range(n):
        if item == items[i]:
            return 1
    return 0


cdef int bsearch(int item, int *items, int n) nogil:
    """binary search for a given item in a sorted array of items"""

    cdef int lo = 0
    cdef int hi = n - 1
    cdef int md

    while lo <= hi:
        md = int(lo + (hi - lo) / 2)
        if items[md] == item:
            return 1
        elif (items[md] < item):
            lo = md + 1
        else:
            hi = md - 1
    return 0


cdef float compute_ui_utility(
    int F,
    int P,
    int Q,
    float[::1] x_uf,
    float[::1] x_if,
    float w_i,
    float[::1] w_if,
    float[::1] v_u,
    float[::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    int x_uf_any,
    int x_if_any
) nogil:

    cdef int f, p, q
    cdef float res = w_i

    for f in range(F):
        # user * item: np.dot(v_u[u], v_i[i])
        res += v_u[f] * v_i[f]

    if x_uf_any:
        for p in range(P):
            if x_uf[p] == 0.0:
                continue
            for f in range(F):
                # user-features * item: np.dot(x_uf[u], np.dot(v_uf, v_i[i]))
                res += x_uf[p] * (v_uf[p, f] * v_i[f])

    if x_if_any:
        for q in range(Q):
            if x_if[q] == 0.0:
                continue
            # item-features: np.dot(x_if[i], w_if)
            res += x_if[q] * w_if[q]
            for f in range(F):
                # item-features * user: np.dot(x_if[i], np.dot(v_if, v_u[u]))
                res += x_if[q] * (v_if[q, f] * v_u[f])

    return res

# -------------------------
# [Python] helper functions
# -------------------------

def assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if):
    """assert all model weights are finite"""

    assert np.isfinite(np.sum(w_i)), "item weights [w_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(w_if)), "item feature weights [w_if] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_u)), "user factors [v_u] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_i)), "item factors [v_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_uf)), "user-feature factors [v_uf] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_if)), "item-feature factors [v_if] are not finite - try decreasing feature/sample_weight magnitudes"


def reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if):
    """calculate the total regularization penalty for all model weights"""

    penalty = 0.0
    penalty += np.sum(alpha * np.square(w_i))
    penalty += np.sum(alpha * np.square(v_u))
    penalty += np.sum(alpha * np.square(v_i))
    penalty += np.sum(beta * np.square(w_if))
    penalty += np.sum(beta * np.square(v_uf))
    penalty += np.sum(beta * np.square(v_if))
    return penalty

# --------------------------------
# [RankFM] core modeling functions
# --------------------------------

def _fit(
    int[:, ::1] interactions,
    float[::1] sample_weight,
    dict user_items,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    float alpha,
    float beta,
    float learning_rate,
    str learning_schedule,
    float learning_exponent,
    int max_samples,
    int epochs,
    bint verbose
):

    #############################
    ### VARIABLE DECLARATIONS ###
    #############################

    # constants
    cdef float MARGIN = 1.0

    # matrix shapes/indicators
    cdef int N, U, I, F, P, Q
    cdef int x_uf_any, x_if_any

    # loop iterators/indices
    cdef int r, u, i, j, f, p, q
    cdef int epoch, row, sampled

    # epoch-specific learning rate and log-likelihood
    cdef float eta, log_likelihood

    # sample weights and (ui, uj) utility scores
    cdef float sw, ut_ui, ut_uj

    # WARP sampling variables
    cdef int min_index
    cdef float pairwise_utility, min_pairwise_utility, multiplier

    # loss function derivatives wrt model weights
    cdef float d_outer
    cdef float d_reg_a = 2.0 * alpha
    cdef float d_reg_b = 2.0 * beta
    cdef float d_w_i = 1.0
    cdef float d_w_j = -1.0
    cdef float d_w_if, d_v_i, d_v_j, d_v_u, d_v_uf, d_v_if

    #######################################
    ### PYTHON SET-UP PRIOR TO TRAINING ###
    #######################################

    # initialize MT random state
    mt.init_genrand(1492)

    # calculate matrix shapes
    N = interactions.shape[0]
    U = v_u.shape[0]
    I = v_i.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # determine whether any user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # create a shuffle index to diversify each training epoch and register as a memoryview to use in NOGIL
    shuffle_index = np.arange(N, dtype=np.int32)
    cdef int[:] shuffle_index_mv = shuffle_index

    # count the total number of items for each user
    items_user = {user: len(items) for user, items in user_items.items()}

    # create c-arrays: number of items and sorted array of items for each user
    cdef int *c_items_user = <int*>malloc(U * sizeof(int))
    cdef int **c_user_items = <int**>malloc(U * sizeof(int*))

    # fill the c-arrays from the P-arrays to use later in NOGIL blocks
    for u in range(U):
        c_items_user[u] = items_user[u]
        c_user_items[u] = <int*>malloc(c_items_user[u] * sizeof(int))
        for i in range(c_items_user[u]):
            c_user_items[u][i] = user_items[u][i]

    ################################
    ### MAIN TRAINING EPOCH LOOP ###
    ################################

    for epoch in range(epochs):

        # set the learning rate for this training epoch
        if learning_schedule == 'constant':
            eta = learning_rate
        elif learning_schedule == 'invscaling':
            eta = learning_rate / pow(epoch + 1, learning_exponent)
        else:
            raise ValueError('unknown [learning_schedule]')

        # re-shuffle the interaction data for each epoch
        np.random.shuffle(shuffle_index)
        log_likelihood = 0.0

        for r in range(N):

            # locate the observed (user, item, sample-weight)
            row = shuffle_index_mv[r]
            u = interactions[row, 0]
            i = interactions[row, 1]
            sw = sample_weight[row]

            # compute the utility score of the observed (u, i) pair
            ut_ui = compute_ui_utility(F, P, Q, x_uf[u], x_if[i], w_i[i], w_if, v_u[u], v_i[i], v_uf, v_if, x_uf_any, x_if_any)

            # WARP sampling loop for the (u, i) pair
            # --------------------------------------

            min_index = -1
            min_pairwise_utility = 1e6

            for sampled in range(1, max_samples + 1):

                # randomly sample an unobserved item (j) for the user
                while True:
                    j = mt.genrand_int32() % I
                    if not lsearch(j, c_user_items[u], c_items_user[u]):
                        break

                # compute the utility score of the unobserved (u, j) pair and the subsequent pairwise utility
                ut_uj = compute_ui_utility(F, P, Q, x_uf[u], x_if[j], w_i[j], w_if, v_u[u], v_i[j], v_uf, v_if, x_uf_any, x_if_any)
                pairwise_utility = ut_ui - ut_uj

                if pairwise_utility < min_pairwise_utility:
                    min_index = j
                    min_pairwise_utility = pairwise_utility

                if pairwise_utility < MARGIN:
                    break

            # set the final sampled negative item index and calculate the WARP multiplier
            j = min_index
            pairwise_utility = min_pairwise_utility
            multiplier = log((I - 1) / sampled) / log(I)
            log_likelihood += log(1 / (1 + exp(-pairwise_utility)))

            # gradient step model weight updates
            # ----------------------------------

            # calculate the outer derivative [d_LL / d_g(pu)]
            d_outer = 1.0 / (exp(pairwise_utility) + 1.0)

            # update the [item] weights
            w_i[i] += eta * (sw * multiplier * (d_outer * d_w_i) - (d_reg_a * w_i[i]))
            w_i[j] += eta * (sw * multiplier * (d_outer * d_w_j) - (d_reg_a * w_i[j]))

            # update the [item-feature] weights
            if x_if_any:
                for q in range(Q):
                    d_w_if = x_if[i, q] - x_if[j, q]
                    w_if[q] += eta * (sw * multiplier * (d_outer * d_w_if) - (d_reg_b * w_if[q]))

            # update all [factor] weights
            for f in range(F):

                # [user-factor] and [item-factor] derivatives wrt [user-factors] and [item-factors]
                d_v_u = v_i[i, f] - v_i[j, f]
                d_v_i = v_u[u, f]
                d_v_j = -v_u[u, f]

                # add [user-features] to [item-factor] derivatives if supplied
                if x_uf_any:
                    for p in range(P):
                        d_v_i += v_uf[p, f] * x_uf[u, p]
                        d_v_j -= v_uf[p, f] * x_uf[u, p]

                # add [item-features] in [user-factor] derivatives if supplied
                if x_if_any:
                    for q in range(Q):
                        d_v_u += v_if[q, f] * (x_if[i, q] - x_if[j, q])

                # update the [user-factor] and [item-factor] weights with the final gradient values
                v_u[u, f] += eta * (sw * multiplier * (d_outer * d_v_u) - (d_reg_a * v_u[u, f]))
                v_i[i, f] += eta * (sw * multiplier * (d_outer * d_v_i) - (d_reg_a * v_i[i, f]))
                v_i[j, f] += eta * (sw * multiplier * (d_outer * d_v_j) - (d_reg_a * v_i[j, f]))

                # update the [user-feature-factor] weights if user features were supplied
                if x_uf_any:
                    for p in range(P):
                        if x_uf[u, p] == 0.0:
                            continue
                        d_v_uf = x_uf[u, p] * (v_i[i, f] - v_i[j, f])
                        v_uf[p, f] += eta * (sw * multiplier * (d_outer * d_v_uf) - (d_reg_b * v_uf[p, f]))

                # update the [item-feature-factor] weights if item features were supplied
                if x_if_any:
                    for q in range(Q):
                        if x_if[i, q] - x_if[j, q] == 0.0:
                            continue
                        d_v_if = (x_if[i, q] - x_if[j, q]) * v_u[u, f]
                        v_if[q, f] += eta * (sw * multiplier * (d_outer * d_v_if) - (d_reg_b * v_if[q, f]))

        # [end epoch]: assert all model weights are finite
        assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if)

        # report the penalized log-likelihood for this training epoch
        if verbose:
            penalty = reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if)
            log_likelihood = round(log_likelihood - penalty, 2)
            print("\ntraining epoch:", epoch)
            print("log likelihood:", log_likelihood)

    # [end training]: free memory of c-arrays before exiting function
    for u in range(U):
        free(c_user_items[u])
    free(c_items_user)
    free(c_user_items)


def _predict(
    float[:, ::1] pairs,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
):

    # declare variables
    cdef int N, P, Q, F
    cdef int x_uf_any, x_if_any
    cdef int row, u, i
    cdef float u_flt, i_flt

    # calculate matrix shapes
    N = pairs.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # determine whether any user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # initialize the output scores vector
    scores = np.empty(N, dtype=np.float32)

    for row in range(N):

        # locate the user/item to score
        u_flt = pairs[row, 0]
        i_flt = pairs[row, 1]

        # set the score to nan if the user or item not found
        if np.isnan(u_flt) or np.isnan(i_flt):
            scores[row] = np.nan
        else:
            # calculate the pointwise utility score for the (u, i) pair
            u, i = int(u_flt), int(i_flt)
            scores[row] = compute_ui_utility(F, P, Q, x_uf[u], x_if[i], w_i[i], w_if, v_u[u], v_i[i], v_uf, v_if, x_uf_any, x_if_any)

    return scores


def _recommend(
    float[::1] users,
    dict user_items,
    int n_items,
    bint filter_previous,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if
):

    # declare variables
    cdef int U, I, P, Q, F
    cdef int x_uf_any, x_if_any
    cdef int row, u, i, s
    cdef float u_flt

    # calculate matrix shapes
    U = users.shape[0]
    I = w_i.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # determine whether any user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # initialize the [UxR] recommendations matrix
    rec_items = np.empty((U, n_items), dtype=np.float32)

    # initialize a temporary buffer to store all item scores for a given user
    item_scores = np.empty(I, dtype=np.float32)
    cdef float[::1] item_scores_mv = item_scores

    for row in range(U):
        u_flt = users[row]
        if np.isnan(u_flt):
            # set the rec item vector to nan if the user not found
            rec_items[row] = np.full(n_items, np.nan, dtype=np.float32)
        else:
            # calculate the scores for all items wrt the current user
            u = int(u_flt)
            for i in range(I):
                item_scores_mv[i] = compute_ui_utility(F, P, Q, x_uf[u], x_if[i], w_i[i], w_if, v_u[u], v_i[i], v_uf, v_if, x_uf_any, x_if_any)

            # get a ranked list of item index positions for the user
            ranked_items = np.argsort(item_scores)[::-1]
            selected_items = np.empty(n_items, dtype=np.float32)

            # select the topN items for each user, optionally skipping previously observed items
            s = 0
            for i in range(I):
                if filter_previous and ranked_items[i] in user_items[u]:
                    continue
                else:
                    selected_items[s] = ranked_items[i]
                    s += 1
                if s == n_items:
                    break

            rec_items[row] = selected_items

    return rec_items

