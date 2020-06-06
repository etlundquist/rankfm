#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

# -----------------------
# [C/Python] dependencies
# -----------------------

from libc.stdlib cimport malloc, free, rand
from libc.math cimport log, exp, pow
cimport cython

import numpy as np

# ---------------------
# [C] utility functions
# ---------------------


cdef void add_vv(int n, float *x, float *y, float *out) nogil:
    """add two vectors and place the result in a third"""

    cdef int i
    for i in range(n):
        out[i] = x[i] + y[i]


cdef void sub_vv(int n, float *x, float *y, float *out) nogil:
    """subtract two vectors and place the result in a third"""

    cdef int i
    for i in range(n):
        out[i] = x[i] - y[i]


cdef float dot_vv(int n, float *x, float *y) nogil:
    """compute the dot product of two vectors"""

    cdef int i
    cdef float res = 0.0
    for i in range(n):
        res += x[i] * y[i]
    return res


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


# --------------------------
# [Python] utility functions
# --------------------------


def assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if):
    """assert all model weights are finite"""

    assert np.isfinite(np.sum(w_i)), "item weights [w_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(w_if)), "item feature weights [w_if] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_u)), "user factors [v_u] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_i)), "item factors [v_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_uf)), "user-feature factors [v_uf] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_if)), "item-feature factors [v_if] are not finite - try decreasing feature/sample_weight magnitudes"


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


# --------------------------------
# [RankFM] core modeling functions
# --------------------------------


def fit(
    int[:, ::1] interactions,
    float[:] sample_weight,
    dict user_items,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[:] w_i,
    float[:] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    float regularization,
    float learning_rate,
    float learning_exponent,
    int max_samples,
    int epochs,
):

    #############################
    ### VARIABLE DECLARATIONS ###
    #############################

    # matrix shapes/indicators
    cdef int N, U, I, P, Q, F
    cdef int x_uf_any, x_if_any

    # loop iterators/indices
    cdef int f, r, u, i, j
    cdef int epoch, row, sampled

    # sample weights, learning rate, log-likelihood
    cdef float sw, eta, log_likelihood

    # WARP sampling parameters
    cdef int margin = 1
    cdef int min_index
    cdef float pairwise_utility, min_pairwise_utility

    # loss function inner derivatives wrt model weights
    cdef float d_con
    cdef float d_reg = 2.0 * regularization
    cdef float d_w_i = 1.0, d_w_j = -1.0
    cdef float d_v_i, d_v_j, d_v_u

    #######################################
    ### PYTHON SET-UP PRIOR TO TRAINING ###
    #######################################

    # calculate matrix shapes
    N = interactions.shape[0]
    U = v_u.shape[0]
    I = v_i.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # create temporary memoryviews to hold intermediate calculation results
    cdef float[:] f_ij = np.empty(F, dtype=np.float32)

    # determine whether user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # create a shuffle index to diversify each training epoch
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

        # set the learning rate and re-set the log-likelihood for this epoch
        eta = learning_rate / pow(epoch + 1, learning_exponent)
        log_likelihood = 0.0

        # re-shuffle the data prior to training
        np.random.shuffle(shuffle_index)

        # begin training loop over observed (u, i) pairs
        for r in range(N):

            # locate the observed user/item/sample-weight
            row = shuffle_index_mv[r]
            u = interactions[row, 0]
            i = interactions[row, 1]
            sw = sample_weight[row]

            # record the index and the pairwise utility of the worst rank reversal
            min_index = -1
            min_pairwise_utility = 1e6

            # start the WARP sampling loop for the (u, i) pair
            for sampled in range(1, max_samples + 1):

                # randomly sample an unobserved item (j) for the user
                while True:
                    j = rand() % I
                    if not lsearch(j, c_user_items[u], c_items_user[u]):
                        break

                # take the vector difference of item factors: f_ij = v_i(i) - v_i(j)
                sub_vv(F, &v_i[i][0], &v_i[j][0], &f_ij[0])

                # calculate the pairwise utility score for the (u, i, j) triplet
                pairwise_utility = 0.0
                pairwise_utility += w_i[i] - w_i[j]
                pairwise_utility += dot_vv(F, &v_u[u][0], &f_ij[0])

                # check to see if this (u, i, j) triple has the worst rank reversal seen so far
                if pairwise_utility < min_pairwise_utility:
                    min_index = j
                    min_pairwise_utility = pairwise_utility

                # check to see if this (u, i, j) triple violates the rank reversal margin
                if pairwise_utility < margin:
                    break

            # set the final sampled negative item index and calculate the WARP multiplier
            j = min_index
            pairwise_utility = min_pairwise_utility
            multiplier = log((I - 1) / sampled) / log(I)
            log_likelihood += log(1 / (1 + exp(-pairwise_utility)))

            # calculate the outer-derivative [d_LL/d_g(pu)]
            d_con = 1.0 / (exp(pairwise_utility) + 1.0)

            # update the item weights with a gradient step
            w_i[i] += eta * (sw * multiplier * (d_con * d_w_i) - (d_reg * w_i[i]))
            w_i[j] += eta * (sw * multiplier * (d_con * d_w_j) - (d_reg * w_i[j]))

            # update the user/item factor weights with a gradient step
            for f in range(F):
                d_v_u = v_i[i][f] - v_i[j][f]
                d_v_i = v_u[u][f]
                d_v_j = -v_u[u][f]
                v_u[u][f] += eta * (sw * multiplier * (d_con * d_v_u) - (d_reg * v_u[u][f]))
                v_i[i][f] += eta * (sw * multiplier * (d_con * d_v_i) - (d_reg * v_i[i][f]))
                v_i[j][f] += eta * (sw * multiplier * (d_con * d_v_j) - (d_reg * v_i[j][f]))

        ##########################
        ### END TRAINING EPOCH ###
        ##########################

        # assert all model weights are finite as of the end of this epoch
        assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if)

        # report the penalized log-likelihood for this training epoch
        penalty = reg_penalty(regularization, w_i, w_if, v_u, v_i, v_uf, v_if)
        log_likelihood = round(log_likelihood - penalty, 2)
        print("\ntraining epoch:", epoch)
        print("log likelihood:", log_likelihood)

    ####################################
    ### END MAIN TRAINING EPOCH LOOP ###
    ####################################

    # free c-arrays memory
    for u in range(U):
        free(c_user_items[u])
    free(c_items_user)
    free(c_user_items)

