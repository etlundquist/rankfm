Quickstart
==========

Let's work through a simple example of fitting a model, generating recommendations, evaluating performance, and assessing some item-item similarities. The data we'll be using here may already be somewhat familiar: you know it, you love it, it's the `MovieLens 1M <https://grouplens.org/datasets/movielens/1m/>`_!

Let's first look at the required shape of the interaction data:

======= =======
user_id item_id    
======= =======
3       233
5       377
8       610
======= =======

It has just two columns: a `user_id` and an `item_id` (you can name these fields whatever you want or use a numpy array instead). Notice that there is no `rating` column - this library is for **implicit feedback** data (e.g. watches, page views, purchases, clicks) as opposed to **explicit feedback** data (e.g. 1-5 ratings, thumbs up/down). Implicit feedback is far more common in real-world recommendation contexts and doesn't suffer from the `missing-not-at-random problem <https://resources.bibblio.org/hubfs/share/2018-01-24-RecSysLDN-Ravelin.pdf>`_ of pure explicit feedback approaches.

Now let's import the library, initialize our model, and fit on the training data:

.. code:: python

  from rankfm.rankfm import RankFM
  model = RankFM(factors=20, loss='warp', max_samples=20, learning_rate=0.1, learning_schedule='invscaling')
  model.fit(interactions_train, epochs=20, verbose=True)

If you set `verbose=True` the model will print the current epoch number as well as the epoch's log-likelihood during training. This can be useful to gauge both computational speed and training gains by epoch. If the log likelihood is not increasing then try upping the `learning_rate` or lowering the (`alpha`, `beta`) regularization strength terms. If the log likelihood is starting to bounce up and down try lowering the `learning_rate` or using `learning_schedule='invscaling'` to decrease the learning rate over time. If you run into overflow errors then decrease the feature and/or sample-weight magnitudes and try upping `beta`, especially if you have a small number of dense user-features and/or item-features. Selecting `BPR` loss will lead to faster training times, but `WARP` loss typically yields superior model performance.

Now let's generate some user-item model scores from the validation data:

.. code:: python

  valid_scores = model.predict(interactions_valid, cold_start='nan')

this will produce an array of real-valued model scores generated using the Factorization Machines model equation. You can interpret it as a measure of the predicted utility of item (i) for user (u). The `cold_start='nan'` option can be used to set scores to `np.nan` for user/item pairs not found in the training data, or `cold_start='drop'` can be specified to drop those pairs so the results contain no missing values.

Now let's generate our topN recommended movies for each user:

.. code:: python

  valid_recs = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='drop')

The input should be a `pd.Series`, `np.ndarray` or `list` of `user_id` values. You can use `filter_previous=True` to prevent generating recommendations that include any items observed by the user in the training data, which could be useful depending on your application context. The result will be a `pd.DataFrame` where `user_id` values will be the index and the rows will be each user's top recommended items in descending order (best item is in column 0):

=======  ====  ====  ====  ====  ====  ====  ====  ==== ====  ====
user_id     0     1     2     3     4     5     6     7    8     9
=======  ====  ====  ====  ====  ====  ====  ====  ==== ====  ====
3        2396  1265   357    34  2858  3175     1  2028   17   356
5         608  1617  1610  3418   590   474   858   377  924  1036
8         589  1036  2571  2028  2000  1220  1197   110  780  1954
=======  ====  ====  ====  ====  ====  ====  ====  ==== ====  ====

Now let's see how the model is performing wrt the included validation metrics evaluated on the hold-out data:

.. code:: python

  from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall

  valid_hit_rate = hit_rate(model, interactions_valid, k=10)
  valid_reciprocal_rank = reciprocal_rank(model, interactions_valid, k=10)
  valid_dcg = discounted_cumulative_gain(model, interactions_valid, k=10)
  valid_precision = precision(model, interactions_valid, k=10)
  valid_recall = recall(model, interactions_valid, k=10)

.. parsed-literal::

  hit_rate: 0.796
  reciprocal_rank: 0.339
  dcg: 0.734
  precision: 0.159
  recall: 0.077

`That's a Bingo! <https://www.youtube.com/watch?v=q5pESPQpXxE>`_

Now let's find the most similar other movies for a few movies based on their embedding representations in latent factor space:

.. code:: python

  # Terminator 2: Judgment Day (1991)
  model.similar_items(589, n_items=10)

.. parsed-literal::

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

`I hope you like explosions... <https://www.youtube.com/watch?v=uENYMZNzg9w>`_

.. code:: python

  # Being John Malkovich (1999)
  model.similar_items(2997, n_items=10)

.. parsed-literal::

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

`Let's get weird... <https://www.youtube.com/watch?v=lIpev8JXJHQ&t=5s>`_

