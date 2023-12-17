from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch as th
import torch.optim as optim

from ...data.containers.tabular_container import TabularContainer
from ...data.iterators.base_iterator import UdaoIterator
from ...utils.interfaces import UdaoInput, UdaoInputShape
from ...utils.logging import logger
from .. import concepts as co
from ..utils.exceptions import NoSolutionError
from .base_solver import SOSolver


def get_default_device() -> th.device:
    return th.device("cuda") if th.cuda.is_available() else th.device("cpu")


class MOGD(SOSolver):
    """MOGD solver for single-objective optimization.

    Performs gradient descent on input variables by minimizing an
    objective loss and a constraint loss.
    """

    @dataclass
    class Params:
        learning_rate: float
        """learning rate of Adam optimizer applied to input variables"""
        weight_decay: float
        """weight decay of Adam optimizer applied to input variables"""
        """TODO: remove"""
        max_iters: int
        """maximum number of iterations for a single local search"""
        patience: int
        """maximum number of iterations without improvement"""
        multistart: int
        """number of random starts for gradient descent"""
        objective_stress: float
        """stress term for objective function"""
        seed: int = 0
        """seed for random number generator"""
        batch_size: int = 1
        """batch size for gradient descent"""
        device: Optional[th.device] = field(default_factory=get_default_device)

        dtype: th.dtype = th.float32

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.lr = params.learning_rate
        self.wd = params.weight_decay
        self.max_iter = params.max_iters
        self.patience = params.patience
        self.multistart = params.multistart
        self.objective_stress = params.objective_stress
        self.seed = params.seed
        self.batch_size = params.batch_size
        self.device = params.device
        self.dtype = params.dtype

    def _get_input_values(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        objective_function: co.ModelComponent,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UdaoInput, UdaoInputShape, Callable[[th.Tensor], TabularContainer]]:
        """Get random values for numeric variables

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        objective_function : co.ModelComponent
            Objective function as a ModelComponent
        input_parameters : Optional[Dict[str, Any]], optional
            Non decision parts of the input, by default None

        Returns
        -------
        Tuple[UdaoInput, UdaoInputShape, Callable[[th.Tensor], TabularContainer]]
            - random values for numeric variables
            - shape of the input
            - function to convert a tensor to a TabularContainer
        """
        numeric_values: Dict[str, np.ndarray] = {}

        for name, variable in numeric_variables.items():
            numeric_values[name] = co.variable.get_random_variable_values(
                variable, self.batch_size
            )

        input_data, iterator = objective_function.process_data(
            input_parameters=input_parameters or {}, input_variables=numeric_values
        )
        make_tabular_container = cast(
            UdaoIterator, iterator
        ).get_tabular_features_container

        input_data_shape = iterator.shape

        return input_data, input_data_shape, make_tabular_container

    def _get_input_bounds(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        objective_function: co.ModelComponent,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UdaoInput, UdaoInput]:
        """Get bounds of numeric variables

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        objective_function : co.ModelComponent
            Objective function as a ModelComponent
        input_parameters : Optional[Dict[str, Any]], optional
            Input parameters, by default None

        Returns
        -------
        Tuple[UdaoInput, UdaoInput]
            Lower and upper bounds of numeric
            variables in the form of a UdaoInput
        """
        lower_numeric_values = {
            name: variable.lower for name, variable in numeric_variables.items()
        }
        upper_numeric_values = {
            name: variable.upper for name, variable in numeric_variables.items()
        }
        lower_input, _ = objective_function.process_data(
            input_parameters=input_parameters,
            input_variables=lower_numeric_values,
        )
        upper_input, _ = objective_function.process_data(
            input_parameters=input_parameters,
            input_variables=upper_numeric_values,
        )
        return lower_input, upper_input

    def _single_start_opt(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        objective: co.Objective,
        constraints: Sequence[co.Constraint],
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float], float]:
        """Perform a single start optimization.
        Categorical variables are fixed to the values in input_parameters.
        (a grid search of categorical variables is performed in solve)
        This is where gradient descent is performed.

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        objective : co.Objective
            Objective to be optimized
        constraints : Sequence[co.Constraint]
            Constraints to be satisfied
        input_parameters : Optional[Dict[str, Any]], optional
            Non decision parts of the input, by default None

        Returns
        -------
        Tuple[float, Dict[str, float], flat]
            - objective value
            - variables
            - best loss value

        Raises
        ------
        Exception
            Either objective or constraints are not ModelComponents
        NoSolutionError
            No valid solution is found
        """
        best_iter = 0
        best_loss = np.inf
        best_obj: Optional[float] = None
        best_feature_input: Optional[th.Tensor] = None
        if not isinstance(objective.function, co.ModelComponent):
            raise Exception("Objective function must be a ModelComponent to use MOGD")
        for constraint in constraints:
            if not isinstance(constraint.function, co.ModelComponent):
                raise Exception(
                    "Constraint functions must be instances of"
                    " ModelComponent to use MOGD"
                )
        # Random numeric variables and their characteristics
        input_data, input_data_shape, make_tabular_container = self._get_input_values(
            numeric_variables, objective.function, input_parameters=input_parameters
        )
        # Bounds of numeric variables
        lower_input, upper_input = self._get_input_bounds(
            numeric_variables, objective.function, input_parameters=input_parameters
        )
        # Indices of numeric variables on which to apply gradients
        mask = th.tensor(
            [i in numeric_variables for i in input_data_shape.feature_input_names]
        )
        grad_indices = th.nonzero(mask, as_tuple=False).squeeze()
        input_vars_subvector = (
            input_data.feature_input[:, grad_indices].clone().detach()
        )
        input_vars_subvector.requires_grad_(True)

        optimizer = optim.Adam([input_vars_subvector], lr=self.lr)

        for i in range(self.max_iter):
            input_data.feature_input[:, grad_indices] = input_vars_subvector
            # Compute objective, constraints and corresponding losses
            obj_output = objective.function.model(input_data)
            objective_loss = self.objective_loss(obj_output, objective)
            constraint_loss = th.zeros_like(objective_loss)

            if constraints:
                const_outputs = [
                    cast(co.ModelComponent, constraint.function).model(input_data)
                    for constraint in constraints
                ]
                constraint_loss = self.constraints_loss(const_outputs, constraints)
            loss = objective_loss + constraint_loss
            sum_loss = th.sum(loss)
            min_loss, min_loss_id = th.min(loss), th.argmin(loss)

            is_within_constraints = constraint_loss[min_loss_id] == 0
            is_within_objective_bounds = self.within_objective_bounds(
                obj_output[min_loss_id].item(), objective
            )
            if (
                min_loss.item() < best_loss
                and is_within_constraints
                and is_within_objective_bounds
            ):
                best_loss = min_loss.item()
                best_obj = obj_output[min_loss_id].item()
                best_feature_input = (
                    input_data.feature_input.cpu()[min_loss_id]
                    .detach()
                    .clone()
                    .reshape(1, -1)
                )
                best_iter = i

            optimizer.zero_grad()
            sum_loss.backward(retain_graph=True)  # type: ignore
            optimizer.step()

            # Update input_vars_subvector with constrained values
            input_vars_subvector.data = th.clip(
                input_vars_subvector.data,
                # use .data to avoid gradient tracking during update
                lower_input.feature_input[0, grad_indices],
                upper_input.feature_input[0, grad_indices],
            )
            if i > best_iter + self.patience:
                break

        if best_obj is not None:
            logger.debug(
                f"Finished at iteration {iter}, best local {objective.name} "
                f"found {best_obj:.5f}"
                f" \nat iteration {best_iter},"
                f" \nwith vars: {best_feature_input}, for "
                f"objective {objective} and constraints {constraints}"
            )

            best_feature_input = cast(th.Tensor, best_feature_input)
            feature_container = make_tabular_container(best_feature_input)
            best_raw_df = objective.function.inverse_process_data(
                feature_container, "tabular_features"
            )
            best_raw_vars = {
                name: best_raw_df[[name]]
                .values.squeeze()
                .tolist()  # turn np.ndarray to float
                for name in numeric_variables
            }
            return best_obj, best_raw_vars, best_loss
        else:
            logger.debug(
                f"Finished at iteration {iter}, no valid {objective.name}"
                f" found for input parameters {input_parameters} with "
                f"objective {objective} and constraints {constraints}"
            )
            raise NoSolutionError

    def solve(self, problem: co.SOProblem) -> Tuple[float, Dict[str, float]]:
        th.manual_seed(self.seed)
        categorical_variables = [
            name
            for name, variable in problem.variables.items()
            if isinstance(variable, co.EnumVariable)
        ]
        numeric_variables = {
            name: variable
            for name, variable in problem.variables.items()
            if isinstance(variable, co.NumericVariable)
        }
        meshed_categorical_vars = self.get_meshed_categorical_vars(problem.variables)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = np.array([0])

        best_loss_list: List[float] = []
        obj_list: List[float] = []
        vars_list: List[Dict] = []
        for _ in range(self.multistart):
            for categorical_cell in meshed_categorical_vars:
                categorical_values = {
                    name: categorical_cell[ind]
                    for ind, name in enumerate(categorical_variables)
                }  # from {id: value} to {name: value}
                fixed_values = {
                    **categorical_values,
                    **(problem.input_parameters or {}),
                }
                try:
                    (
                        obj_pred,
                        best_raw_vars,
                        best_loss,
                    ) = self._single_start_opt(
                        numeric_variables=numeric_variables,
                        input_parameters=fixed_values,
                        objective=problem.objective,
                        constraints=problem.constraints or [],
                    )
                except NoSolutionError:
                    continue
                else:
                    best_loss_list.append(best_loss)
                    obj_list.append(obj_pred)
                    vars_list.append(best_raw_vars)
        if not obj_list:
            raise NoSolutionError("No valid solutions and variables found!")
        idx = np.argmin(best_loss_list)
        vars_cand = vars_list[idx]
        if vars_cand is not None:
            obj_cand = obj_list[idx]
            if obj_cand is None:
                raise Exception(f"Unexpected objs_list[{idx}] is None.")
        else:
            raise NoSolutionError("No valid solutions and variables found!")

        return obj_cand, vars_cand

    ##################
    ## _loss        ##
    ##################
    def constraints_loss(
        self, constraint_values: List[th.Tensor], constraints: Sequence[co.Constraint]
    ) -> th.Tensor:
        """
        compute loss of the values of each constraint function fixme: double-check

        Parameters
        ----------
        constraint_values : List[th.Tensor]
            values of each constraint function
        constraints : Sequence[co.Constraint]
            constraint functions

        Returns
        -------
        th.Tensor
            loss of the values of each constraint function

        """

        # vars: a tensor
        # get loss for constraint functions defined in the problem setting
        total_loss = th.zeros_like(constraint_values[0])
        for i, (constraint_value, constraint) in enumerate(
            zip(constraint_values, constraints)
        ):
            constraint_violation = th.zeros_like(constraint_values[0])
            if constraint.upper is not None and constraint.lower is not None:
                if constraint.upper == constraint.lower:
                    constraint_violation = th.abs(constraint_value - constraint.upper)
                else:
                    normed_constraint = (constraint_value - constraint.lower) / (
                        constraint.upper - constraint.lower
                    )
                    constraint_violation = th.where(
                        (normed_constraint < 0) | (normed_constraint > 1),
                        (normed_constraint - 0.5),
                        0,
                    )
            elif constraint.lower is not None:
                constraint_violation = th.relu(constraint.lower - constraint_value)
            elif constraint.upper is not None:
                constraint_violation = th.relu(constraint_value - constraint.upper)
            total_loss += (
                constraint_violation**2
                + constraint.stress * (constraint_violation > 0).float()
            )

        return total_loss

    def objective_loss(
        self, objective_value: th.Tensor, objective: co.Objective
    ) -> th.Tensor:
        """Compute the objective loss for a given objective value:
        - if no bounds are specified, use the squared objective value
        - if both bounds are specified, use the squared normalized
        objective value if it is within the bounds, otherwise
        add a stress term to a squared distance to middle of the bounds

        Parameters
        ----------
        objective_value : th.Tensor
            Tensor of objective values
        objective : co.Objective
            Objective function

        Returns
        -------
        th.Tensor
            Tensor of objective losses

        Raises
        ------
        NotImplementedError
            If only one bound is specified for the objective

        """
        loss = th.zeros_like(objective_value)  # size of objective_value ((bs, 1) ?)
        if objective.upper is None and objective.lower is None:
            loss = (
                th.sign(objective_value) * (objective_value**2) * objective.direction
            )
        elif objective.upper is not None and objective.lower is not None:
            norm_cst_obj_pred = (objective_value - objective.lower) / (
                objective.upper - objective.lower
            )  # scaled
            loss = th.where(
                (norm_cst_obj_pred < 0) | (norm_cst_obj_pred > 1),
                (norm_cst_obj_pred - 0.5) ** 2 + self.objective_stress,
                norm_cst_obj_pred * objective.direction,
            )
        else:
            raise NotImplementedError("Objective with only one bound is not supported")
        return loss

    ##################
    ## _get (vars)  ##
    ##################

    def get_meshed_categorical_vars(
        self, variables: Dict[str, co.Variable]
    ) -> Optional[np.ndarray]:
        """
        Get combinations of all categorical (binary, enum) variables

        Parameters
        ----------
        variables : Dict[str, co.Variable]
            Variables to be optimized

        Returns
        -------
        Optional[np.ndarray]
            Combinations of all categorical variables
            of shape (n_samples, n_vars)
        """
        cv_value_list = [
            variable.values
            for variable in variables.values()
            if isinstance(variable, co.EnumVariable)
        ]
        if not cv_value_list:
            return None
        meshed_cv_value_list = [x_.reshape(-1, 1) for x_ in np.meshgrid(*cv_value_list)]
        meshed_cv_value = np.concatenate(meshed_cv_value_list, axis=1)
        return meshed_cv_value

    ##################
    ## _check       ##
    ##################

    # check violations of objective value var_ranges
    # reuse code in UDAO
    @staticmethod
    def within_objective_bounds(obj_value: float, objective: co.Objective) -> bool:
        """
        check whether violating the objective value var_ranges
        :param pred_dict: dict, keys are objective names,
        values are objective values
        :param obj_bounds: dict, keys are objective names,
        values are lower and upper var_ranges of each objective value
        :return: True or False
        """
        within_bounds = True
        if objective.upper is not None:
            within_bounds = obj_value <= objective.upper
        if objective.lower is not None:
            within_bounds = within_bounds and obj_value >= objective.lower
        return within_bounds
