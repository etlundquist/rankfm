
import pytest
import numpy as np
import pandas as pd

from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall

# ------------------------------
# create sample data for testing
# ------------------------------

# training interactions
# ---------------------

# train interactions as pd.dataframe of [int]
intx_train_pd_int = pd.DataFrame([
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 6),
    (3, 3), (3, 6), (3, 4)
], columns=['user_id', 'item_id'], dtype=np.int32)

# train interactions as pd.dataframe of [str]
intx_train_pd_str = pd.DataFrame([
    ('X', 'A'), ('X', 'C'), ('X', 'E'),
    ('Y', 'A'), ('Y', 'B'), ('Y', 'F'),
    ('Z', 'C'), ('Z', 'F'), ('Z', 'D')
], columns=['user_id', 'item_id'])

# train interactions as a np.ndarray
intx_train_np = np.array([
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 6),
    (3, 3), (3, 6), (3, 4)
])

# train interactions with an erroneous extra rating column
intx_train_pd_rating = pd.DataFrame([
    (1, 1, 5), (1, 3, 2), (1, 5, 3),
    (2, 1, 2), (2, 2, 1), (2, 6, 4),
    (3, 3, 3), (3, 6, 4), (3, 4, 5)
], columns=['user_id', 'item_id', 'rating'], dtype=np.int32)

# user features
# -------------

# user features data as a pd.dataframe in the correct format
uf_pd_good = pd.DataFrame([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62)
], columns=['user_id', 'bin_1', 'bin_2', 'int', 'cnt'])

# user features as a np.ndarray in the correct format
uf_np_good = np.array([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62)
])

# user features with a missing ID column
uf_no_id = pd.DataFrame([
    (0, 1, 5, 3.14),
    (1, 0, 6, 2.72),
    (0, 0, 4, 1.62)
], columns=['bin_1', 'bin_2', 'int', 'cnt'])

# user features with a string column
uf_str_cols = pd.DataFrame([
    (1, 0, 1, "A", 3.14),
    (2, 1, 0, "B", 2.72),
    (3, 0, 0, "C", 1.62)
], columns=['user_id', 'bin_1', 'bin_2', 'str', 'cnt'])

# item features
# -------------

# item features data as a pd.dataframe in the correct format
if_pd_good = pd.DataFrame([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
    (4, 1, 1, 3, 1.05),
    (5, 1, 0, 6, 0.33),
    (6, 0, 0, 0, 0.00)
], columns=['item_id', 'bin_1', 'bin_2', 'int', 'cnt'])

# item features as a np.ndarray in the correct format
if_np_good = np.array([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
    (4, 1, 1, 3, 1.05),
    (5, 1, 0, 6, 0.33),
    (6, 0, 0, 0, 0.00)
])

# item features with a missing ID column
if_no_id = pd.DataFrame([
    (0, 1, 5, 3.14),
    (1, 0, 6, 2.72),
    (0, 0, 4, 1.62),
    (1, 1, 3, 1.05),
    (1, 0, 6, 0.33),
    (0, 0, 0, 0.00)
], columns=['bin_1', 'bin_2', 'int', 'cnt'])

# item features with a string column
if_str_cols = pd.DataFrame([
    (1, 0, 1, "A", 3.14),
    (2, 1, 0, "B", 2.72),
    (3, 0, 0, "C", 1.62),
    (4, 1, 1, "A", 1.05),
    (5, 1, 0, "F", 0.33),
    (6, 0, 0, "G", 0.00)
], columns=['item_id', 'bin_1', 'bin_2', 'str', 'cnt'])

# ------------------------------
# test basic model functionality
# ------------------------------

# model fitting
# -------------

params_good = [
    (intx_train_pd_int,       None,       None),
    (intx_train_pd_str,       None,       None),
    (intx_train_np,           None,       None),
    (intx_train_pd_int, uf_pd_good,       None),
    (intx_train_pd_int,       None, if_pd_good),
    (intx_train_pd_int, uf_pd_good, if_pd_good),
    (intx_train_pd_int, uf_np_good, if_np_good),
]

@pytest.mark.parametrize("interactions, user_features, item_features", params_good)
def test__fit__good(interactions, user_features, item_features):
    """assert that the model can be successfully fit on the input data"""

    model = RankFM(factors=2)
    model.fit(interactions, user_features, item_features, epochs=2, verbose=True)
    assert model.is_fit


def test__fit__bad__rating_col():
    """ensure that having more than 2 columns in the interaction data causes an assertion failure"""

    with pytest.raises(AssertionError):
        model = RankFM(factors=2)
        model.fit(intx_train_pd_rating)


def test__fit__bad__uf_no_id():
    """ensure that the [user_features] contains a [user_id] column"""

    with pytest.raises(KeyError):
        model = RankFM(factors=2)
        model.fit(intx_train_pd_int, user_features=uf_no_id)

def test__fit__bad__uf_str_cols():
    """ensure that the [user_features] does not contain string columns"""

    with pytest.raises(ValueError):
        model = RankFM(factors=2)
        model.fit(intx_train_pd_int, user_features=uf_str_cols)


def test__fit__bad__if_no_id():
    """ensure that the [item_features] contains a [item_id] column"""

    with pytest.raises(KeyError):
        model = RankFM(factors=2)
        model.fit(intx_train_pd_int, item_features=if_no_id)

def test__fit__bad__if_str_cols():
    """ensure that the [item_features] does not contain string columns"""

    with pytest.raises(ValueError):
        model = RankFM(factors=2)
        model.fit(intx_train_pd_int, item_features=if_str_cols)






# model = RankFM(factors=2)
# model.fit(interactions=intx_train_pd_int, user_features=uf_no_id)
