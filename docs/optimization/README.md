## **1. Weighted Sum**

Weighted Sum (WS) method calls the method `weighted_sum` in the packages `optimization.moo`. 
Within the method, it calls two solvers `random_sampler` and `grid_search` of package `optimization.solver`.

Before running the example, it is necessary to configure the problem settings.

### **1.1 Problem setting**
- ##### **Configuration file**:
   
   Please provide a `.json` file for the configurations used in problem setting, e.g. file `"examples/optimization/ws_random/heuristic_closed_form/configs.json"`
   
   ```yaml
   "moo_algo"         -> the name of MOO algorithm,`
  
   "solver"           -> the name of solver,`
  
   "variables"        -> a list of variables with name, type, lower and upper bounds`
  
   "objectives"       -> a list of objectives with name, optimize_trend (e.g. min/max) and type`
  
   "additional_params"-> parameters for a specific algorithm with its solver

   NOTE: if the bounds of variables is inifite, please set it to a concrete number rather than setting it as `inf`
- ##### **Define functions of objectives and constraints**:
   Please define your optimization problem in the file `utils/optimization/functions_def.py`

### **1.2 Run Weighted Sum**
   1). set the python path

   `export PYTHONPATH=~/your_path_to/UDAO2022`

   2). run the following command
   ```yaml
   # an example for weighted sum algorithm with Grid-Search solver
   # problem settings are defined in the config.json file and utils/optimization/functions_def.py
   python examples/optimization/ws_grid/heuristic_closed_form/ws_grid_hcf.py -c examples/optimization/ws_grid/heuristic_closed_form/configs.json```
