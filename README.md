# RankFM

[![PyPI version](https://badge.fury.io/py/rankfm.svg)](https://badge.fury.io/py/rankfm)
[![CircleCI](https://circleci.com/gh/etlundquist/rankfm.svg?style=shield)](https://circleci.com/gh/etlundquist/rankfm)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

RankFM is a python implementation of the general Factorization Machines model class described in [Rendle 2010](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) adapted for collaborative filtering recommendation/ranking problems with implicit feedback user-item interaction data. It uses [Bayesian Personalized Ranking (BPR)](https://arxiv.org/pdf/1205.2618.pdf) and a variant of [Weighted Approximate-Rank Pairwise (WARP)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.587.3946&rep=rep1&type=pdf) loss to learn model weights via Stochastic Gradient Descent (SGD). It can (optionally) incorporate individual training sample weights and/or user/item auxiliary features to augment the main interaction data for model training.

The core training/prediction/recommendation methods are written in [Cython](https://cython.org/). This makes it possible to scale to millions of users, items, and interactions. Designed for ease-of-use, RankFM accepts both `pd.DataFrame` and `np.ndarray` inputs. You do not have to convert your data to `scipy.sparse` matrices or re-map user/item identifiers to matrix indexes prior to use - RankFM internally maps all user/item identifiers to zero-based integer indexes, but always converts its outputs back to the original user/item identifiers from your data, which can be arbitrary (non-zero-based, non-consecutive) integers or even strings.

In addition to the familiar `fit()`, `predict()`, `recommend()` methods, RankFM includes additional utilities `similiar_users()` and `similar_items()` to find the most similar users/items to a given user/item based on latent factor space embeddings. A number of popular recommendation/ranking evaluation metric functions have been included in the separate `evaluation` module to streamline model tuning and validation. See the **Quickstart** section below to get started, and the `/examples` folder for more in-depth jupyter notebook walkthroughs with several popular open-source data sets.

This package is currently under active development and should not yet be considered fully stable. The core functionality is in place and working, but has not yet been rigorously tested against a wide variety of real-world data sets, modeling objectives, edge cases, user errors, etc. If you do find a problem or have suggestions for improvement please let me know!

---
### Dependencies
* Python 3.6+
* numpy >= 1.15
* pandas >= 0.24
* Cython >= 0.29

### Installation

#### Prerequisites

To install RankFM you will first need the [GNU Compiler Collection (GCC)](https://gcc.gnu.org/). This is a free open-source C/C++ compiler that will build RankFM's Cython extensions into platform-specific Python extension modules (e.g. `_rankfm.cpython-37m-darwin.so`).

On Mac OSX I recommend installing via [Homebrew](https://brew.sh/):
```
brew install gcc
```
On Linux (e.g. AWS EC2) you can just use your system's built-in package manager:
```
sudo yum install gcc
```
To check whether GCC has been installed successfully simply run:
```
gcc --version
```

#### Package Installation

You can install the latest published version from PyPI using `pip`:
```
pip install rankfm
```
Or alternatively install the current development build directly from GitHub:
```
pip install git+https://github.com/etlundquist/rankfm.git#egg=rankfm
```

It's highly recommended that you use an [Anaconda](https://www.anaconda.com/) base environment to ensure that all core numpy C extensions and linear algebra libraries have been installed and configured correctly. Anaconda: it just works.

### Quickstart
Let's work through a simple example of fitting a model, generating recommendations, evaluating performance, and assessing some item-item similarities. The data we'll be using here may already be somewhat familiar: you know it, you love it, it's the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)!

Let's first look at the required shape of the interaction data:

| user_id | item_id |
|---------|---------|
| 3       | 233     |
| 5       | 377     |
| 8       | 610     |

It has just two columns: a `user_id` and an `item_id` (you can name these fields whatever you want or use a numpy array instead). Notice that there is no `rating` column - this library is for **implicit feedback** data (e.g. watches, page views, purchases, clicks) as opposed to **explicit feedback** data (e.g. 1-5 ratings, thumbs up/down). Implicit feedback is far more common in real-world recommendation contexts and doesn't suffer from the [missing-not-at-random problem](https://resources.bibblio.org/hubfs/share/2018-01-24-RecSysLDN-Ravelin.pdf) of pure explicit feedback approaches.

Now let's import the library, initialize our model, and fit on the training data:
```python
from rankfm.rankfm import RankFM
model = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
model.fit(interactions_train, epochs=20, verbose=True)
# NOTE: this takes about 30 seconds for 750,000 interactions on my 2.3 GHz i5 8GB RAM MacBook
```
If you set `verbose=True` the model will print the current epoch number as well as the epoch's log-likelihood during training. This can be useful to gauge both computational speed and training performance by epoch. If the log likelihood is not increasing then try upping the `learning_rate` or lowering the `regularization`. If the log likelihood is starting to bounce up and down try lowering the `learning_rate` or using `learning_schedule='invscaling'` to decrease the learning rate over time. Selecting `BPR` loss will lead to faster training times, but `WARP` loss typically yields superior model performance.

Now let's generate some user-item model scores from the validation data:
```python
valid_scores = model.predict(interactions_valid, cold_start='nan')
```
this will produce an array of real-valued model scores generated using the Factorization Machines model equation. You can interpret it as a measure of the predicted utility of item (i) for user (u). The `cold_start='nan'` option can be used to set scores to `np.nan` for user/item pairs not found in the training data, or `cold_start='drop'` can be specified to drop those pairs so the results contain no missing values.

Now let's generate our topN recommended movies for each user:
```python
valid_recs = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='drop')
```
The input should be a `pd.Series`, `np.ndarray` or `list` of `user_id` values. You can use `filter_previous=True` to prevent generating recommendations that include any items observed by the user in the training data, which could be useful depending on your application context. The result will be a `pd.DataFrame` where `user_id` values will be the index and the rows will be each user's top recommended items in descending order (best item is in column 0):

|   |    0|    1|    2|    3|    4|    5|    6|    7|   8|    9|
|---|-----|-----|-----|-----|-----|-----|-----|-----|----|-----|
|3  | 2396| 1265|  357|   34| 2858| 3175|    1| 2028|  17|  356|
|5  |  608| 1617| 1610| 3418|  590|  474|  858|  377| 924| 1036|
|8  |  589| 1036| 2571| 2028| 2000| 1220| 1197|  110| 780| 1954|

Now let's see how the model is performing wrt the included validation metrics evaluated on the hold-out data:
```python
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall

valid_hit_rate = hit_rate(model, interactions_valid, k=10)
valid_reciprocal_rank = reciprocal_rank(model, interactions_valid, k=10)
valid_dcg = discounted_cumulative_gain(model, interactions_valid, k=10)
valid_precision = precision(model, interactions_valid, k=10)
valid_recall = recall(model, interactions_valid, k=10)
```
```
hit_rate: 0.796
reciprocal_rank: 0.339
dcg: 0.734
precision: 0.159
recall: 0.077
```
[That's a Bingo!](https://www.youtube.com/watch?v=q5pESPQpXxE)

Now let's find the most similar other movies for a few movies based on their embedding representations in latent factor space:
```python
# Terminator 2: Judgment Day (1991)
model.similar_items(589, n_items=10)
```
```
2571                       Matrix, The (1999)
1527                Fifth Element, The (1997)
2916                      Total Recall (1990)
3527                          Predator (1987)
780             Independence Day (ID4) (1996)
1909    X-Files: Fight the Future, The (1998)
733                          Rock, The (1996)
1376     Star Trek IV: The Voyage Home (1986)
480                      Jurassic Park (1993)
1200                            Aliens (1986)
```
[I hope you like explosions...](https://www.youtube.com/watch?v=uENYMZNzg9w)

```python
# Being John Malkovich (1999)
model.similar_items(2997, n_items=10)
```
```
2599           Election (1999)
3174    Man on the Moon (1999)
2858    American Beauty (1999)
3317        Wonder Boys (2000)
223              Clerks (1994)
3897      Almost Famous (2000)
2395           Rushmore (1998)
2502       Office Space (1999)
2908     Boys Don't Cry (1999)
3481      High Fidelity (2000)
```
[Let's get weird...](https://www.youtube.com/watch?v=lIpev8JXJHQ&t=5s)

---
That's all for now. To see more in-depth worked examples in jupyter notebook format head to the `/examples` folder. Be sure to check back for added functionality and updated PyPI releases in the near future. There's more to come - stay tuned...

---
![under construction](./images/UnderConstruction.png)

