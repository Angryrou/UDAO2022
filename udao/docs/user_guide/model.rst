===============================
Modeling the objective function
===============================

The objective function is what we want to minimize in the optimization module.
Here we aim at training a machine learning model from the training data prepared in :doc:`Data processing <../user_guide/data_processing>`, as the objective function to minimize.

The case of query plans
-----------------------

To model a latency function from a query plan graph and features, we provide two components:

* Embedders that embed the query plan graph and features into a vector space, based on the :py:class:`~udao.model.embedders.base_graph_embedder.BaseGraphEmbedder` class.
* Regressors that take tabular features concatened with the query plan embedding to output the predicted latency. The :py:class:`~udao.model.regressors.mlp.MLP` implements an MLP regressor.
