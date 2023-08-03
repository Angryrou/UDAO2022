* [Current code base](#current-code-base)
    * [Dataset](#dataset)
    * [Model](#model)
    * [Optimization](#optimization)
* [Coding work to be done](#Coding-work-to-be-done)
    * [Data Processing Module](#data-processing-module)
    * [Modeling Module](#modeling-module)
    * [Optimization Module](#optimization-module)
* [The end-to-end usage of `udao`](#the-end-to-end-usage-of-udao)
    * [Input/Output Diagram](#input-output-diagram)
    * [Desire Example](#desire-example)

## Current code base

### Dataset

1. [A TPCH trace](https://github.com/Angryrou/UDAO-release/wiki/Trace-Release) with 100K data points (released). The query plan information is maintained in a graph data structure, while other features and objectives are stored in a tabular DataFrame.
    ```python
    # check examples/traces/spark-tpch-100-traces/main.py
    data_header = "examples/traces/spark-tpch-100-traces/data"
    graph_data = PickleUtils.load(data_header, "graph_data.pkl")
    tabular_data = PickleUtils.load(data_header, "tabular_data.pkl")
    
    # sample output
    print(graph_data.keys())
    # dict_keys(['all_ops', 'dgl_dict'])
    
    print(tabular_data.keys())
    # dict_keys(['ALL_COLS', 'COL_MAP', 'df'])
    
    print(tabular_data["COL_MAP"])
    #  {'META_COLS': ['id', 'q_sign', 'template', 'start_timestamp', 'latency'],
    #  'CH1_FEATS': ['dgl_id'],
    #  'CH2_FEATS': ['input_mb', 'input_records', 'input_mb_log', 'input_records_log'],
    #  'CH3_FEATS': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8'],
    #  'CH4_FEATS': ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 's1', 's2', 's3', 's4'],
    #  'OBJS': ['latency']}
    ```
    - A graph structure has two types of features for a query plan: graph topology and operator features.
    - When the operator feature is just the operator type, many data points have the same graph features. Therefore, we saved all distinct graph topologies in a DGLGraph list and indexed them with `dgl_id` in the tabular data. 
    
2. A dataset with more sophisticated operator features:
   - When the operator features go beyond the operator type, data points become more diverse in the feature space. However, saving all the distinct graph structures is not memory-friendly. Therefore, we separate the storage of the graph topologies and operator features into three steps:
      1. We first maintain all the distinct graph structures in a DGLGraph list. 
      2. For each graph structure (g), we store all the operator features of g in a 3D array `node_feat_group` with the dimensions as follows: `[# of data points with the structure g, g.number_of_nodes, dimension of the operator features]`.
      3. We then define the `'CH1_FEATS': ['dgl_id', 'vid']`, where `dgl_id` represents the index of the graph topology and `vid` represents the index of the data point in the `node_feat_group`.

3. [Separate code files](https://github.com/Angryrou/UDAO2022/blob/trace-parsing-and-modeling-spark/examples/trace/spark-parser) to construct graph data and tabular data from the raw traces. However, this part can be dropped in our Python library since different datasets may require different construction methods. Instead, we plan to use the same dataset structure to unify the training and optimization processes.


### Model

1. We have implemented various models' architecture in `PyTorch`, including MLP, GTN, GCN, GATv2, and more. Find these implementations [here](https://github.com/Angryrou/UDAO2022/tree/trace-parsing-and-modeling-spark/model/architecture).

2. We have designed a complete [pipeline](https://github.com/Angryrou/UDAO2022/blob/7d9eff4ad8bb2b276b2f27522f6b67d9e3b9c205/utils/model/utils.py#L1039) for model training, which includes the following steps:
    - Model setup
    - Data setup
    - Training component setup (e.g., training optimizer, log, loss function, etc.)
    - Training (iteratively training the model)

### Optimization

1. [An initial Multi-Objective Optimization (MOO) module](https://github.com/Angryrou/UDAO-release/blob/main/docs/optimization/README.md) (released)
    ```python
    ├── __init__.py
    ├── moo
    │   ├── __init__.py
    │   ├── base_moo.py
    │   ├── evolutionary.py
    │   ├── generic_moo.py # (entry point)
    │   ├── progressive_frontier.py
    │   └── weighted_sum.py
    ├── model
    │   ├── __init__.py
    │   └── base_model.py
    └── solver
        ├── __init__.py
        ├── base_solver.py
        ├── grid_search.py
        ├── mogd.py
        └── random_sampler.py
    ```
    - Currently, the `model` submodule in this MOO module is isolated from our modeling part. One solution to connect them is to inherit all the built-in models from the modeling part to the `BaseModel` in `base_model.py`.
    - To demonstrate the capabilities of the MOO module, we provide 3-4 separate examples that include closed-form models, GPR models, and tiny neural networks. 

## Coding work to be done

We aim to integrate our code into a Python library called "udao," making it accessible for users to install and utilize through a simple `pip install udao` command. The `udao` library is designed to offer three core modules:

1. Data Processing Module (`from udao import dataset`)

2. Modeling Module (`from udao import model`)

3. Optimization Module (`from udao import moo`)

We summarize the coding work into three categories. 

### Data Processing Module

1. Design and implement a `Dataset` class to support multi-channel feature inputs with two types of data structures and necessary meta information. The `Dataset` class maintains
    - *graph data*, maintained in `dgl.DGLGraph`
        + The query plan in a Graph topology, e,g.,`g = dgl.graph((src_ids, dst_ids))`
        + The operator features in `g.ndata["feat"]`
    - *tabular data*, maintained in `pandas.DataFrame`
        + The graph id (e.g., with column ["gid", "fid"])
        + The input meta information (e.g., with columns `["input_records", "input_bytes"]`)
        + The machine system states (e.g., with columns `["m1", "m2"]`)
        + The configuration (e.g., with columns `["k1", "k2"]`)
    - the meta information 
        + `num_of_data`: the total number of data points.
        + `all_tfeat_cols`: the column names of all tabular features.
        + `all_ofeat_cols`: the column names of the operator features.
        + `tfeat_dict`: a dict of the column names of different tabular feature types. E.g., `{"I": ["col1", "col2"], "M": ["m1", "m2"], "C": ["k1", "k2"]]}`
        + `ofeat_dict`: a dict of feature indices of the operator features in the graph. E.g., `{"type": [0], "cbo": [1, 2], "predicates": [3, 4, 5]}`
    - other APIs
        + An API to declare the features to be used for training. E.g., `dataset.Dataset.declare(ofeats=["col1", "col2", "k1", "k2"], tfeats=["type", "cbo"])`
        + An internal API to fetch the corresponding graph data given a data point in the tabular row. 

2. Implement 2-3 built-in datasets by integrating our existing datasets. 

5. Provide a toy example of adding a customized Dataset.

6. Implement an API to auto-load a built-in or customized dataset, e.g., `d = dataset.load("TPCH")`

7. Implement an API for the data preprocessing pipeline, including
    - train/val/test split
    - drop unnecessary columns
    - convert categorical features to the dummy vector or integer
    - feature augment (if needed)
    - feature normalization

### Modeling Module

1. Design an abstract class `ModelWrapper` to provide the necessary APIs to seamlessly integrate a model with MOO.
    - An abstract method `def initialize(self, *args)` to initialize the model
    - An abstract method `def load(self, *args)` to set the model weights by either loading from the given knowledge or fitting from scratch.
        - (Optional) An abstract method `def fit(self, dataset, loss, hps, *args)` to train the model with the provided dataset, loss function, and the hyperparameters for training.
    - An abstract method `def predict(self, obj, config, *args)` to obtain the value of the target objective given a configuration
    - Other abstract methods if needed.    

2. Implement built-in models, including
    - AVG-MLP (averaging the operator features to embed the query plan)
    - GTN-MLP (use GTN to embed the query plan)
    
3. Implement built-in model wrappers by integrating our built-in models.

5. An API to fetch a built-in `ModelWrapper`, e.g., `m = model.fetch("udao-GTN")`


### Optimization Module

1. Refactoring the current code base to have
    - a class named `Variables` to wrap each variable in the optimization problem.
    - a class named `Configuration` to define the set of all tunable variables.
    - a class named `Objective` to define an objective and specifies the optimization direction.
    - a class named `Constraint` to define specific constraints in the optimization problem.
    - a class named `Solution` to include a configuration and the corresponding objective values (a set of objective values).
    - a class named `ParetoOptimalSet` that encompasses several Pareto-optimal solutions. 

2. Implement a pipeline to support end-to-end optimization with the following procedures.
    - Define variables (supporting integer, float, a float vector, etc.)
    - Define objectives (including the predictive function for the objective and the optimization direction)
    - Define constraints
    - Run the optimization
    - Recommend Pareto-optimal solutions
    - Select one solution with the weighted Utopia-nearest method or the user-defined preferences.

3. Refactor other utils functionality for the module
    - the MOO recommendation methods: including weighted Utopia-nearest method or user-defined preferences.
    - Visualization of the Pareto-optimal solutions (2D and 3D)
   
## The end-to-end usage of `udao`

### Input/Output Diagram
The I/O of `udao` is as follows and we shall be able to use the `moo` module to solve the MOO problem given the dataset and user-defined optimization problem.

<img src="https://github.com/Angryrou/UDAO2022/assets/8079921/ed24c8c9-ca68-46b5-b2b2-b37c424a0080" width="70%">

### A Desired Example
An desired way to use `moo` package

```python
from udao import dataset, model, moo

# 1. Dataset definition. 
d = dataset.load("TPCH") # or declare a customized dataset 

# 2. Problem definition.
# (1) define the variables inside `Configuration` based on spark_knob.json
x = moo.Configuration(meta="spark_knob.json", source=d)  
# (2) define the objectives 
o1 = moo.Objective(name="latency", direction="-1")
o2 = moo.Objective(name="cost", direction="-1")
# (3) define the constraints if any
c = moo.Constraint(func=[]) # in our case, there is not external constraints

# 3. Solving details
# (1) the model choice: 
mw = model.fetch("udao-GTN") # fetch a built-in ModelWrapper (mw) 
o1.set_predictive_function(func=mw.predict, obj_name="latency")
o2.set_predictive_function(func=mw.predict, obj_name="cost")
# (2) the algorithm and solver for MOO
moo_algo = "pf-ap"
moo_solver = "mogd"
# (3) the return preferences
return_type = "PO-set"

# Calling the model to solve an MOO problem
po_solutions = moo.solve(
    objs=[o1, o2], 
    configuration=x, 
    constraints=c, 
    algo=moo_algo,
    solver=moo_solver,
    return_type=return_type)

```