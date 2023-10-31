===================
Data processing API
===================

This section is dedicated to the API documentation of the data processing module of UDAO.
The data processing module is responsible for processing the raw data in view of:

* training a machine learning model to describe the objective function
* process the raw data to be used as input for the optimization module along with the trained model.

It contains the following submodules:

.. toctree::
   :maxdepth: 2

   data_handler
   iterators
   containers
   extractors
   predicate_embedders
   preprocessors
