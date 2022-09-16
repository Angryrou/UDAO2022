**1. Weighted Sum**
    
Weighted Sum (WS) method calls the method `weighted_sum` in the packages `optimization.moo`. 
Within the method, it calls two solvers `random_sampler` and `grid_search` of package `optimization.solver`.

The `examples/optimization/ws_grid_hcf.py` and `examples/optimization/ws_random_hcf.py` show concrete examples on how to use WS, 
where hcf means the objective and constraint functions of this example is based on the heuristic closed form.
You can define your Multi-Objective Optimization (MOO) problem by following the main steps in the example. By default, we assume a minimization problem.

The following is the main steps in `examples/optimization/ws_grid_hcf.py`:

1). define the variable settings with range and type

    n_vars = 2
    lower = np.array([[-20], [-np.inf]])
    upper = np.array([[np.inf], [20]])
    bounds = np.hstack([lower, upper])
    var_types = ["float", "float"]

2). create a class which extends `WeightedSum` API, and rewrite the objective and constrait functions to get objective values and constraint violations.
    
    from optimization.moo.weighted_sum import WeightedSum
    class ExampleWSWithGridSearch(WeightedSum):
        def __init__(self, gs_params, n_objs, debug):
            super().__init__(gs_params, n_objs, debug)
        def _obj_function(self, vars, obj):
            ...
        def _const_function(self, vars):
            ...

3). set necessary parameters for grid_search/random_sampler

    ## for grid_search
    other_params = {}
    other_params["inner_solver"] = "grid_search"
    other_params["num_ws_pairs"] = 10
    gs_params = {}
    gs_params["n_grids_per_param"] = 10
    other_params["gs_params"] = gs_params

4). initialize the example class, and run the solve method

    example = ExampleWSWithGridSearch(other_params, n_objs=2, debug=False)
    po_objs, po_vars = example.solve(bounds, var_types, n_objs=2)

5). print and plot the final results

    print(po_objs)
    print(po_vars)
    if po_objs is not None:
        example.plot_po(po_objs)

**NOTE:** 

1. Currently, in each solver, I put a test code for the correctness of the methods used.
2. The variable are set within the method of `def _get_input(self, bounds, var_types)`. 
   For the variables with infinite range, currently I set it as 10 just for the example to return feasilbe 
   solutions within a narrow range. In theory, it should be a very large number (e.g. `1e20`).






