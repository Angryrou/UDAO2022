## Overview of optimization package
`optimization` package includes `optimization.moo` package and `optimization.solver` package. `optimization.moo` package provides APIs to access all Multi-Objective Optimization (MOO) algorithms.
`optimization.solver` package includes APIs to support solving Single-objective Optimization (SO) problems and Constrainted single-objective Optimization (CO) problem, which is called by MOO methods internally in `optimization.moo` package.

`optimization.moo` package provides an entry point API `optimization.moo.generic_moo.GenericMOO` to solve MOO problems. It specifies input parameters for an optimization problem and a MOO algorithm. Based on the choice parameter, the appropriate MOO algorithms run internally which are described in details later in the README.md.

```yaml
   # optimization.moo.generic_moo.GenericMOO
   class GenericMOO:
    def __init__(self, moo_algo, solver, obj_names, obj_funcs, opt_types, const_funcs, const_types, var_types, var_bounds, add_confs):
         # common input paramters for MOO problems
        self.obj_names = obj_names     # objective names
        self.obj_funcs = obj_funcs     # objective functions
        self.opt_types = opt_types     # optimization types (minimization/maximization)
        self.const_funcs = const_funcs # constraint functions
        self.const_types = const_types # constraint types ("<=" or "<", e.g. g1(x1, x2, ...) - c <= 0)
        self.var_types = var_types     # variable types (float, integer, binary)
        self.var_bounds = var_bounds   # lower and upper bounds of variables
      
        # specify MOO algorithm, its solver and the specific input parameters the choice of MOO algorithm and its solver
        assert moo_algo in ["weighted_sum", "progressive_frontier", "evolutionary", "mobo", "normalized_normal_constraint"]
        self.moo_algo = moo_algo       # the name of MOO algorithms
        self.solver = solver           # the name of solvers
        self.add_confs = add_confs     # the parameters required by the specified MOO algorithm and solver

    def solve(self):
        # solve MOO problems internally
        if self.moo_algo == "weighted_sum":
            ...
        elif self.moo_algo == 'progressive_frontier':
            ...
        elif self.moo_algo == 'evolutionary':
            ...
        elif self.moo_algo == "mobo":
            ...
        elif self.moo_algo == "normalized_normal_constraint":
            ...
        else:
            raise NotImplementedError
        # return Pareto solutions with the corresponding variables
        return ...
```

## How to run MOO
### **Problem setting**
- ##### **Configuration file**:
   
   A`.json` file is necessary for the configurations used in problem setting, e.g. file `"examples/optimization/ws_random/heuristic_closed_form/configs.json"`
   
   ```yaml
   "moo_algo"         -> the name of the MOO algorithm,` 
   "solver"           -> the name of the solver,` 
   "variables"        -> a list of variables with name, type, lower and upper bounds` 
   "objectives"       -> a list of objectives with name, optimize_trend (e.g. min/max) and type`
   "constraints"      -> a list of constraints with name, type (e.g. "<=", "<")
   "additional_params"-> parameters for a specific algorithm with its solver

   NOTE: if the bounds of variables is inifite, please set it to a concrete number rather than setting it as `inf`
- ##### **Define functions of objectives and constraints**:
   The use-defined optimization problem is set in the file `utils/optimization/functions_def.py`.
The functions of objectives and constraints can either be represented as heuristic closed forms or predictive models (e.g. based on Neural Network).


### **Run MOO**
- First parse the configuration file to obtain input parameters, and then call `optimization.moo.generic_moo.GenericMOO` to return Pareto solutions and the corresponding variables.

```yaml
# get input parameters
moo_algo, solver, var_types, var_bounds, obj_names, opt_types, const_types, add_params = ConfigsParser().parse_details()
# call the entry point API, e.g. for a constrained 2D MOO problem
moo = GenericMOO(moo_algo, solver, obj_names, obj_funcs=[func_def.obj_func1, func_def.obj_func2], opt_types=opt_types,
                 const_funcs=[func_def.const_func1, func_def.const_func2], const_types=const_types, var_types=var_types, var_bounds=var_bounds,
                 add_confs=add_params)
# return Pareto solutions and the corresponding variables
po_objs, po_vars = moo.solve()
```
## **APIs in Optimization package**
The following shows a brief description of APIs in `optimization.moo` and `optimization.solver` packages.

```yaml
   # APIs in moo package (MOO algorithms)
   optimization.moo.generic_moo.GenericMOO                    -> the entry point of all moo algorithms,`
   optimization.moo.based_moo.BaseMOO                         -> the base API includes abstract methods, and all APIs of MOO algorithms extend this API
   optimization.moo.weighted_sum.WeightedSum                  -> API for the Weighted Sum method`
   optimization.moo.progressive_frontier.ProgressiveFrontier  -> API for the Progressive Frontier method`
   optimization.moo.evolutionary.EVO                          -> API for the Evolutionary (NSGA-II) method
   optimization.moo.mobo.MOBO                                 -> API for the Multi-Objective Bayesian Optimization method
   optimization.moo.normalized_normal_constraint.NNC          -> API for the Normalized Normal Constraint method
   
   # APIs in solver package, called internally by MOO algorithms
   optimization.solver.base_solver.BaseSolver                 -> the base API includes abstract methods, and all solver APIs extend this API 
   optimization.solver.grid_search.GridSearch                 -> API for the Grid Search 
   optimization.solver.random_sampler.RandomSampler           -> API for the Random Sampler 
   optimization.solver.mogd.MOGD                              -> API for the Multi-Objective Gradient Descent
   ```
## **MOO Algrithms with Examples**
### **Weighted Sum**

Weighted Sum (WS) method adds different weights on different objectives to show preferences or importance over all objectives. It transforms a MOO problem into a SO. 

In the packages `optimization.moo`, it calls the method `weighted_sum`. 
Within the method, it supports to call the solver `random_sampler` or `grid_search`.

#### **Run Weighted Sum Example**
   The following example calls Weighted Sum algorithm with solver Grid-Search. The functions of objectives and constraints are represented by the heuristic closed form.
   
   1). problem settings shown in 
      `"examples/optimization/ws_grid/heuristic_closed_form/configs.json"` and `utils/optimization/functions_def.py`

   2). set the python path
   `export PYTHONPATH=~/your_path_to/UDAO2022`

   3). run the following command
   ```yaml
   # an example for weighted sum algorithm with Grid-Search solver
   # problem settings are defined in the config.json file and utils/optimization/functions_def.py
   python examples/optimization/ws_grid/heuristic_closed_form/ws_grid_hcf.py -c examples/optimization/ws_grid/heuristic_closed_form/configs.json
```

### **Evolutionary Algorithm**

In the packages `optimization.moo`, it calls the method `evolutionary`. 
Within the method, it calls the library Platypus [[1]] with NSGA-II algorithm. The library is installed by `pip install platypus-opt==1.0.4`.
#### **Run EVO Example**

  The following example calls EVO(NSGA-II) algorithm. The functions of objectives and constraints are represented by the heuristic closed form.
   
   1). problem settings shown in 
      `"examples/optimization/evo/heuristic_closed_form/configs.json"` and `utils/optimization/functions_def.py`

   2). set the python path

   `export PYTHONPATH=~/your_path_to/UDAO2022`

   3). run the following command
   ```yaml
   # an example for EVO(NSGA-II) algorithm
   # problem settings are defined in the config.json file and utils/optimization/functions_def.py
   python examples/optimization/evo/heuristic_closed_form/evo_hcf.py -c examples/optimization/evo/heuristic_closed_form/configs.json
   ```
[1]: https://github.com/Project-Platypus/Platypus/tree/docs