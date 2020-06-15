Welcome to RankFM's Documentation!
==================================

RankFM is a python implementation of the general Factorization Machines model class described in `Rendle 2010 <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_ adapted for collaborative filtering recommendation/ranking problems with implicit feedback user-item interaction data. It uses `Bayesian Personalized Ranking (BPR) <https://arxiv.org/pdf/1205.2618.pdf>`_ and a variant of `Weighted Approximate-Rank Pairwise (WARP) <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.587.3946&rep=rep1&type=pdf>`_ loss to learn model weights via Stochastic Gradient Descent (SGD). It can (optionally) incorporate individual training sample weights and/or user/item auxiliary features to augment the main interaction data for model training.

The core training/prediction/recommendation methods are written in `Cython <https://cython.org/>`_. This makes it possible to scale to millions of users, items, and interactions. Designed for ease-of-use, RankFM accepts both `pd.DataFrame` and `np.ndarray` inputs. You do not have to convert your data to `scipy.sparse` matrices or re-map user/item identifiers to matrix indexes prior to use - RankFM internally maps all user/item identifiers to zero-based integer indexes, but always converts its outputs back to the original user/item identifiers from your data, which can be arbitrary (non-zero-based, non-consecutive) integers or even strings.

In addition to the familiar `fit()`, `predict()`, `recommend()` methods, RankFM includes additional utilities `similiar_users()` and `similar_items()` to find the most similar users/items to a given user/item based on latent factor space embeddings. A number of popular recommendation/ranking evaluation metric functions have been included in the separate `evaluation` module to streamline model tuning and validation.

Dependencies
------------

* Python 3.6+
* numpy >= 1.15
* pandas >= 0.24

Installation
------------

Prerequisites
^^^^^^^^^^^^^

To install RankFM's C extensions you will need the `GNU Compiler Collection (GCC) <https://gcc.gnu.org/>`_. Check to see whether you already have it installed:

.. code:: bash

  gcc --version

If you don't have it already you can easily install it using `Homebrew <https://brew.sh/>`_ on OSX or your default linux package manager:

.. code:: bash

  # OSX
  brew install gcc

  # linux
  sudo yum install gcc

  # ensure [gcc] has been installed correctly and is on the system PATH
  gcc --version

Package Installation
^^^^^^^^^^^^^^^^^^^^

You can install the latest published version from PyPI using `pip`:

.. code:: bash

  pip install rankfm

Or alternatively install the current development build directly from GitHub:

.. code:: bash
  
  pip install git+https://github.com/etlundquist/rankfm.git#egg=rankfm


