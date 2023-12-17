====================================
Optimizing the objective function(s)
====================================

Once we have a trained model for our objective function, we can use it to optimize the objective function.
This is done in the `optimization` module.

Defining a problem
------------------
You can define either a single objective problem or a multi-objective problem, using :py:class:`~udao.optimization.concepts.problem.SOProblem` or :py:class:`~udao.optimization.concepts.problem.MOProblem` respectively.
In both cases, you need to define the following:

* The objective function(s) to optimize
* The constraints
* The variables to optimize
* The fixed input parameters

Single objective problem
~~~~~~~~~~~~~~~~~~~~~~~~
For a single objective problem, you can use an :py:class:`~udao.optimization.soo.base_solver.SOSolver` to optimize the objective function.
The solver will return the optimal value of the objective function and the optimal values of the variables.

Multi-objective problem
~~~~~~~~~~~~~~~~~~~~~~~
For a multi-objective problem, you can use an :py:class:`~udao.optimization.moo.base_moo.MOSolver` to optimize the objective function.

In both cases, you can define the solver and its parameters, and then call solver.solve() to solve the problem.


Defining a solver
-----------------

Single objective solver
~~~~~~~~~~~~~~~~~~~~~~~
Several single objective solvers are available in the :py:mod:`~udao.optimization.soo` module.
They all inherit from :py:class:`~udao.optimization.soo.base_solver.SOSolver`.
You can define your own solver by inheriting from :py:class:`~udao.optimization.soo.base_solver.SOSolver` and implementing the :py:meth:`~udao.optimization.soo.base_solver.SOSolver.solve` method.

Multi-objective solver
~~~~~~~~~~~~~~~~~~~~~~
Several multi-objective solvers are available in the :py:mod:`~udao.optimization.moo` module.
They all inherit from :py:class:`~udao.optimization.moo.base_moo.MOSolver`.
You can define your own solver by inheriting from :py:class:`~udao.optimization.moo.base_moo.MOSolver` and implementing the :py:meth:`~udao.optimization.moo.base_moo.MOSolver.solve` method.
Several multi-objective solvers need to be provided with a single objective solver. You can use any single objective solver that inherits from :py:class:`~udao.optimization.soo.base_solver.SOSolver`.

Putting it all together
-----------------------
Here is an example of how to define a problem and solve it::

    input_parameters = { ... }
     def n_cores(
        input_variables: concepts.InputVariables,
        input_parameters: concepts.InputParameters = None,
    ) -> th.Tensor:
        return th.tensor((input_variables["k3"]) * input_variables["k1"])

    problem = concepts.MOProblem(
        objectives=[
            concepts.Objective(
                name="latency",
                direction_type="MIN",
                function=concepts.ModelComponent(
                    data_processor=data_processor, model=model
                ),
            ),
            concepts.Objective(
                name="cloud_cost", direction_type="MIN", function=n_cores
            ),
        ],
        variables={
            "k1": concepts.IntegerVariable(2, 16),
            "k2": concepts.IntegerVariable(2, 5),
            "k3": concepts.IntegerVariable(4, 10),
        },
        input_parameters=input_parameters,
        constraints=[],
    )
    mogd = MOGD(
        MOGD.Params(
            learning_rate=0.1,
            weight_decay=0.1,
            max_iters=100,
            patience=10,
            seed=0,
            multistart=10,
            objective_stress=0.1,
            batch_size=10,
        )
    )

    mo_solver = SequentialProgressiveFrontier(
        solver=mogd,
        params=SequentialProgressiveFrontier.Params(),
    )

    solution = mo_solver.solve(problem)
