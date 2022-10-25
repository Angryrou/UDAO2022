## Optimization
* [Multi-Objective Optimization Problem](#multi-objective-optimization-problem)
* [Quick Start](#quick-start)
* [Overview of optimization package](#overview-of-optimization-package)
* [How to run MOO](#how-to-run-moo)
    * [Problem setup](#problem-setup)
    * [Run MOO](#run-moo)
* [APIs in Optimization package](#apis-in-optimization-package)
* [MOO Algorithms](#moo-algorithms)
* [Examples](#examples)
  * [Heuristic Closed Form](#heuristic-closed-form)
  * [Gaussian Process Regressor](#gaussian-process-regressor)
  * [Neural Network](#neural-network)
* [TODOs in the next release](#todos-in-the-next-release)

## Multi-Objective Optimization Problem

The Multi-Objective Optimization (MOO) problem could be represented as follows, which includes `k` objectives to be optimized, where along with `m` inequality and `e` equality constraints. The variables **x** are set with lower and upper bounds by $\mathbf{l}$ and $\mathbf{u}$.

$$ \mathop{\rm{Minimize}}\limits_{\rm{\bf{x}}} \\ \\ \mathbf{F(\mathbf{x}}) = [\mathit{F}_1(\mathbf{x}), \mathit{F}_2(\mathbf{x}), ..., \mathit{F}_k(\mathbf{x})]^\mathit{T}$$ 

&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 
subject to   $\\ g_j(\mathbf{x}) \leq 0, \\ j = 1, 2, ..., m$

$$ &ensp; &ensp; h_l(\mathbf{x})=0, \\ l = 1, 2, ..., e $$

$$ \mathbf{l} \le \mathbf{x} \le \mathbf{u} &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp;$$

Generally, optimization of objectives is minimization or maximization. 
Constraints include inequality and equaility types.
Common types of variables include `FLOAT`, `INTEGER`, `BINARY` and `ENUM`, where `BINARY` only includes values 0 and 1, `ENUM` type is for multiple (2+) categorical values.

## Quick Start
The following shows an MOO problem example (Binh and Korn function [[1], [2]]) with 2 objectives, 2 constraints and 2 variables.

$$ \mathop{\rm{Minimize}}\limits_{\rm{\bf{x}}} \\ \\ \mathit{F}_1(\mathbf{x}) = 4 {x_1}^2 + 4 {x_2}^2 $$

$$ \mathit{F}_2(\mathbf{x}) = (x_1 - 5)^2 - (x_2 - 5)^2 $$

&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 
subject to   $\\ (x_1 - 5)^2 + x_2^2 \leq 25 $

$$ (x_1 - 5)^2 + (x_2 + 5)^2 \geq 7.7 &ensp; &ensp;$$

$$ &ensp; &ensp; 0 \le x_1 \le 5, 0 \le x_2 \le 3 $$

The optimization package allows users to define their problems from Python. 
For the details of how to set up problems and how the APIs works internally, please see sections of [how to run MOO](#how-to-run-moo) and [APIs in Optimization package](#apis-in-optimization-package) in the later content.

Here is an [example](../../examples/optimization/heuristic_closed_form/ws.py) for solving the above problem by the Weighted Sum method with the `grid_search` solver. 

```bash
export PYTHONPATH=$PWD # export PYTHONPATH=~/your_path_to/UDAO2022
python examples/optimization/heuristic_closed_form/ws.py -c examples/optimization/heuristic_closed_form/configs/ws_grid_search.json

# output
# 
# Pareto solutions:
# [[136.           4.        ]
#  [111.46168758   4.43107846]
#  [ 91.87184981   5.59422508]
#  [ 76.49586777   7.30578512]
#  [ 55.11196287  11.28420868]
#  [ 36.07134342  16.54943324]
#  [ 21.91174746  22.37910939]
#  [ 11.69498623  28.74223293]
#  [  4.75380386  35.77349741]
#  [  1.22116206  42.49604225]
#  [  0.          50.        ]]
# Variables:
# [[5.         3.        ]
#  [4.34343434 3.        ]
#  [3.73737374 3.        ]
#  [3.18181818 3.        ]
#  [2.62626263 2.62311558]
#  [2.12121212 2.12562814]
#  [1.66666667 1.64321608]
#  [1.21212121 1.20603015]
#  [0.75757576 0.7839196 ]
#  [0.4040404  0.37688442]
#  [0.         0.        ]]
```

The output includes the Pareto solutions and the corresponding variables. The `Pareto solutions` is stacked by a set of Pareto solutions with each column representing the value of one objective.
The `Variables` is stacked by a set of configurations with each column representing the value of one input parameter.
In the above example, we got 11 Pareto solutions and the first Pareto solution is `(136, 4)` in the 2D objective space, corresponding to the 2D variable `(x_1, x_2) = (5, 3)`

The results are also shown in the figure below, where the blue points are the Pareto solutions returned by the Weighted Sum method with te `grid_search` solver in the 2D objective space.

![img_1.png](img_1.png)

## Overview of optimization package
The `optimization` package includes the `optimization.moo` package and the `optimization.solver` package. The `optimization.moo` package provides APIs to access all Multi-Objective Optimization (MOO) methods.
The `optimization.solver` package is called by MOO methods internally in `optimization.moo` package. 

The `optimization.moo` package provides an entry point API `optimization.moo.generic_moo.GenericMOO` to solve MOO problems. It specifies input parameters for an optimization problem and an MOO algorithm. Based on the choice parameter, the appropriate MOO algorithms run internally which are described in details later.

```python
class GenericMOO:

    def __init__(self):
        pass

    def problem_setup(self, obj_names: list, obj_funcs: list, opt_types: list, const_funcs: list, const_types: list,
                      var_types: list, var_ranges: list):
        '''
        setup common input paramters for MOO problems
        :param obj_names: list, objective names
        :param obj_funcs: list, objective functions
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=" "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_ranges: ndarray(n_vars, ), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
        :return:
        '''
        ...

    def solve(self, moo_algo: str, solver: str, add_params: list):
        '''
        solve MOO problems internally by different MOO algorithms
        :param moo_algo: str, the name of moo algorithm
        :param solver: str, the name of solver
        :param add_params: list, the parameters required by the specified MOO algorithm and solver
        :return: po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                 po_vars: ndarray(n_solutions, n_vars), corresponding variables of Pareto solutions
        '''
        ...
        return po_objs, po_vars
```

## How to run MOO

### Problem setup

1. create a directory under `example/optimization/<model>`, e.g., `example/optimization/heuristic_closed_form/`.

2. add a **configuration file** to set up all the parameters for the MOO method. E.g., [Here](../../examples/optimization/heuristic_closed_form/configs/ws_grid_search.json) is 
the configuration file for the example in the [Quick Start](#quick-start).   
  
    ```json
    {
      "moo_algo": "weighted_sum", /* name of MOO algorithms */
      "solver": "grid_search", /* name of the solver */
      "variables": [
        {
          "name": "v1",
          "type": "FLOAT",
          "min": 0,
          "max": 5
        },
        {
          "name": "v2",
          "type": "FLOAT",
          "min": 0,
          "max": 3
        }
      ],
      "objectives": [
        {
          "name": "obj_1",
          "optimize_trend": "MIN", /* to minimize the objective  */
          "type": "FLOAT"
        },
        {
          "name": "obj_2",
          "optimize_trend": "MIN",
          "type": "FLOAT"
        }
      ],
      "constraints": [
        {
          "name": "g1",
          "type": "<="
        },
        {
          "name": "g2",
          "type": ">="
        }
      ],
      "additional_params":
        {
          "ws_steps": 0.1, /* the weight of one objective changes as [0, 0.1, ..., 1]  */
          "solver_params": [100, 200] /* the number of grids per variable  */
        }
    }
    ```
    NOTE: if the bounds of variables is infinity, please set it to a concrete number rather than setting it as `inf`

3. define the functions of objectives and constraints. The functions need to be subdifferentiable, e.g., a close-form formula or a Neural Network model. 
The [Quick Start](#quick-start) example uses [Binh and Korn function][2] provided by our [package](../../examples/optimization/heuristic_closed_form/model.py).

### Run MOO

To run the MOO over the defined problem, construct a python file to invoke `optimization.moo.generic_moo.GenericMOO` with 
the pre-defined functions for the objectives and constraints, and run it with the configuration file.

```bash
export PYTHONPATH=$PWD
# python example/optimization/<model>/<moo-method>.py -c <configuration-file>
python examples/optimization/heuristic_closed_form/ws.py -c examples/optimization/heuristic_closed_form/configs/ws_grid_search.json
```

## APIs in optimization package

The following shows a tree structure of APIs in `optimization` package, where `moo` and `solver` are two packages for MOO algorithms and solvers respectively;
`model` provides a general API for user to inherit.
```bash
├── __init__.py
├── moo
│   ├── __init__.py
│   ├── base_moo.py
│   ├── evolutionary.py
│   ├── generic_moo.py
│   ├── mobo.py
│   ├── normalized_normal_constraint.py
│   ├── progressive_frontier.py
│   └── weighted_sum.py
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

Within the `moo` package, the `generic_moo` provides the entry point of all moo algorithms. The `base_moo` is the base API includes abstract methods, and all APIs of MOO algorithms extend this API. 
`weighted_sum`, `progressive_frontier`, `evolutionary`, `mobo`(Multi-Objective Bayesian Optimization (MOBO), `normalized_normal_constraint` are MOO algorithms supported in the `optimization` package.

Within the `solver` package, the `base_solver` provides the base API includes abstract methods, and all solver APIs extend this API.
`grid_search`, `random_sampler`, `mogd` (Multi-Objective Gradient Descent (MOGD)) are solvers supported in the `optimization` package.

Within the `model` package, the `base_model` provides the base API includes abstract methods used in predictive models.

## MOO Algorithms

The package provides MOO algorithms including:

- [Weighted Sum (WS)][3] method adds different weights on different objectives to show the preferences or importance over all objectives. It transforms an MOO problem into a single-objective optimization problem. Our weighted sum works with the solver `grid_search` and `random_sampler`.
- [todo] Progressive Frontier (PF) method ...
- [todo] Evolutionary (Evo) algorithm ...


## Examples

We provide three examples to do MOO in the 2D objective space, including
- when two objectives are both in the form of heuristic closed form (HCF)
- when two objectives are both from gaussian process regressors (GPR)
- when one objective is a neural network (NN) and the other is HCF.

### Heuristic Closed Form

When the predictive functions (and constraints) are in the 2D objective space,  
[heuristic closed form (Binh and Korn function)](../../examples/optimization/heuristic_closed_form/model.py), 
we show the execution code and results of different MOO methods with solvers below.
 
```bash
export PYTHONPATH=$PWD

# 1. WS 
# WS with the grid_search solver
python examples/optimization/heuristic_closed_form/ws.py -c examples/optimization/heuristic_closed_form/configs/ws_grid_search.json
# WS with the random_sampler solver 
python examples/optimization/heuristic_closed_form/ws.py -c examples/optimization/heuristic_closed_form/configs/ws_random_sampler.json

# [todo] 2. PF

# [todo] ...
```

[TODO] a row of figures, each drawing the PF points in the 2D objective space returned by one MOO method. 

### Gaussian Process Regressor

When the predictive functions (and constraints) are 
[GPR models](../../examples/optimization/gaussian_process_regressor/model.py) in the 2D objective space, 
we show the execution code and results of different MOO methods with solvers below.

```bash
export PYTHONPATH=$PWD

# 1. WS 
# WS with the grid_search solver
python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_grid_search.json
# WS with the random_sampler solver 
python examples/optimization/gaussian_process_regressor/ws.py -c examples/optimization/gaussian_process_regressor/configs/ws_random_sampler.json

# [todo] 2. PF

# [todo] ...
```  

[TODO] a row of figures, each drawing the PF points in the 2D objective space returned by one MOO method. 

### Neural Network

When the predictive functions (and constraints) include 
[NN models](../../examples/optimization/neural_network/model.py) in the 2D objective space (one NN and one HCF), 
we show the execution code and results of different MOO methods with solvers below.

```bash
export PYTHONPATH=$PWD

# 1. WS 
# WS with the grid_search solver
python examples/optimization/neural_network/ws.py -c examples/optimization/neural_network/configs/ws_grid_search.json
# WS with the random_sampler solver 
python examples/optimization/neural_network/ws.py -c examples/optimization/neural_network/configs/ws_random_sampler.json

# [todo] 2. PF

# [todo] ...
```   

[TODO] a row of figures, each drawing the PF points in the 2D objective space returned by one MOO method. 

## TODOs in the next release
1. support additional variable types, such as the assignment matrix where each entry is a non-negative integer, and the sum of each row is given.
2. support a more sophisticated weighted sum approach. The current method only supports uniform weight picking for 2D and 3D. we need ways to randomly generate weights for 2+ objectives.


[1]: https://web.archive.org/web/20190801183649/https://pdfs.semanticscholar.org/cf68/41a6848ca2023342519b0e0e536b88bdea1d.pdf
[2]: https://en.wikipedia.org/wiki/Test_functions_for_optimization#cite_note-Binh97-5
[3]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.404.9971&rep=rep1&type=pdf