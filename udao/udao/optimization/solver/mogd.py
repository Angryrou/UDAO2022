from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch as th
import torch.optim as optim
from torch.multiprocessing import Pool

from ...data.containers.tabular_container import TabularContainer
from ...data.iterators.base_iterator import UdaoIterator
from ...utils.interfaces import UdaoInput, UdaoInputShape
from ...utils.logging import logger
from ..concepts import EnumVariable, NumericVariable, Variable
from ..concepts.constraint import Constraint, ModelConstraint
from ..concepts.objective import ModelObjective, Objective
from ..concepts.variable import get_random_variable_values
from ..utils.exceptions import NoSolutionError
from .base_solver import BaseSolver

SEED = 0
DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32
NOT_FOUND_ERROR = "no valid configuration found"


def get_default_device() -> th.device:
    return th.device("cuda") if th.cuda.is_available() else th.device("cpu")


class MOGD(BaseSolver):
    @dataclass
    class Params:
        learning_rate: float
        weight_decay: float
        max_iters: int
        patient: int
        multistart: int
        processes: int
        stress: float
        seed: int
        batch_size: int = 1
        device: Optional[th.device] = field(default_factory=get_default_device)
        dtype: th.dtype = th.float32

    def __init__(self, params: Params) -> None:
        """
        initialize solver
        :param mogd_params: dict, parameters used in solver
        """
        super().__init__()
        self.lr = params.learning_rate
        self.wd = params.weight_decay
        self.max_iter = params.max_iters
        self.patient = params.patient
        self.multistart = params.multistart
        self.process = params.processes
        self.objective_stress = params.stress
        self.seed = params.seed
        self.device = params.device
        self.dtype = params.dtype
        self.batch_size = params.batch_size

    def _get_input_values(
        self,
        numeric_variables: Dict[str, NumericVariable],
        objective: ModelObjective,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UdaoInput, UdaoInputShape, Callable[[th.Tensor], TabularContainer]]:
        numeric_values: Dict[str, np.ndarray] = {}

        for name, variable in numeric_variables.items():
            numeric_values[name] = get_random_variable_values(variable, self.batch_size)

        input_data, iterator = objective.process_data(
            input_non_decision=input_parameters, input_variables=numeric_values
        )
        make_tabular_container = cast(
            UdaoIterator, iterator
        ).get_tabular_features_container

        input_data_shape = iterator.shape

        return input_data, input_data_shape, make_tabular_container

    def _get_input_bounds(
        self,
        numeric_variables: Dict[str, NumericVariable],
        objective: ModelObjective,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UdaoInput, UdaoInput]:
        lower_numeric_values = {
            name: [variable.lower] for name, variable in numeric_variables.items()
        }
        upper_numeric_values = {
            name: [variable.upper] for name, variable in numeric_variables.items()
        }
        lower_input, _ = objective.process_data(
            input_non_decision=input_parameters, input_variables=lower_numeric_values
        )
        upper_input, _ = objective.process_data(
            input_non_decision=input_parameters, input_variables=upper_numeric_values
        )
        return lower_input, upper_input

    def _single_start_opt(
        self,
        numeric_variables: Dict[str, NumericVariable],
        objective: ModelObjective,
        constraints: Sequence[ModelConstraint],
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float], float]:
        best_iter = 0
        best_loss = np.inf
        best_obj: Optional[float] = None
        best_feature_input: Optional[th.Tensor] = None

        # Random numeric variables and their characteristics
        input_data, input_data_shape, make_tabular_container = self._get_input_values(
            numeric_variables, objective, input_parameters=input_parameters
        )
        # Bounds of numeric variables
        lower_input, upper_input = self._get_input_bounds(
            numeric_variables, objective, input_parameters=input_parameters
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
            obj_output = objective.model(input_data)
            objective_loss = self.objective_loss(obj_output, objective)
            constraint_loss = th.zeros_like(objective_loss)
            if constraints:
                const_outputs = [
                    constraint.model(input_data) for constraint in constraints
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
            if i > best_iter + self.patient:
                break

        if best_obj is not None:
            logger.debug(
                f"Finished at iteration {iter}, best local {objective.name} "
                f"found {best_obj:.5f}"
                f" \nat iteration {best_iter},"
                f" \nwith vars: {best_feature_input}"
            )

            best_feature_input = cast(th.Tensor, best_feature_input)
            feature_container = make_tabular_container(best_feature_input)
            best_raw_df = objective.inverse_process_data(
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
                f" found for input parameters {input_parameters}"
            )
            raise NoSolutionError

    def solve(
        self,
        objective: Objective,
        variables: Dict[str, Variable],
        constraints: Optional[Sequence[Constraint]] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        if not isinstance(objective, ModelObjective):
            raise Exception("Objective must be a ModelObjective to use MOGD")
        if not isinstance(constraints, Sequence[ModelConstraint]):
            raise Exception(
                "Constraints must be instances of ModelConstraint to use MOGD"
            )

        th.manual_seed(self.seed)
        categorical_variables = [
            name
            for name, variable in variables.items()
            if isinstance(variable, EnumVariable)
        ]
        numeric_variables = {
            name: variable
            for name, variable in variables.items()
            if isinstance(variable, NumericVariable)
        }
        meshed_categorical_vars = self.get_meshed_categorical_vars(variables)

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
                fixed_values = {**categorical_values, **(input_parameters or {})}
                try:
                    (
                        obj_pred,
                        best_raw_vars,
                        best_loss,
                    ) = self._single_start_opt(
                        numeric_variables=numeric_variables,
                        input_parameters=fixed_values,
                        objective=objective,
                        constraints=constraints or [],
                    )
                except NoSolutionError:
                    continue
                else:
                    best_loss_list.append(best_loss)
                    obj_list.append(obj_pred)
                    vars_list.append(best_raw_vars)
        if not obj_list:
            raise Exception("No valid solutions and variables found!")
        idx = np.argmin(best_loss_list)
        vars_cand = vars_list[idx]
        if vars_cand is not None:
            obj_cand = obj_list[idx]
            if obj_cand is None:
                raise Exception(f"Unexpected objs_list[{idx}] is None.")
        else:
            obj_cand, vars_cand = None, None

        return obj_cand, vars_cand

    def optimize_constrained_so_parallel(
        self,
        wl_id: str,
        objective_name: str,
        cell_list: list,
        batch_size: int = 1,
    ) -> List[Tuple[Optional[float], Optional[Dict[str, float]]]]:
        """
        solve the single objective optimization
        constrained by objective values parallelly
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj: str, name of objective to be optimized
        :param cell_list: list, each element is a dict to indicate
            the var_ranges of objective values
        :return:
                ret_list: list,
                    each element is a solution
                    tuple with size 2) with objective values (tuple[0], list)
                    and variables (tuple[1], ndarray(1, n_vars))
        """
        th.manual_seed(self.seed)

        # generate the list of input parameters for constraint_so_opt
        arg_list = [
            (
                wl_id,
                objective_name,
                obj_bounds_dict,
                batch_size,
            )
            for obj_bounds_dict in cell_list
        ]
        if th.cuda.is_available():
            th.multiprocessing.set_start_method("spawn", force=True)

        # call self.constraint_so_opt parallely
        with Pool(processes=self.process) as pool:
            ret_list = pool.starmap(self.solve, arg_list)
        return ret_list

    ##################
    ## _loss        ##
    ##################
    def constraints_loss(
        self, constraint_values: List[th.Tensor], constraints: Sequence[ModelConstraint]
    ) -> th.Tensor:
        """
        compute loss of the values of each constraint function fixme: double-check
        :param wl_id: str, workload id, e.g. '1-7'
        :param vars: tensor ((bs, n_vars) or (n_vars, )),
            variables, where bs is batch_size
        :return: const_loss: tensor (Tensor:()),
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
        self, objective_value: th.Tensor, objective: ModelObjective
    ) -> th.Tensor:
        loss = th.zeros_like(objective_value)  # size of objective_value ((bs, 1) ?)
        if objective.upper is None and objective.lower is None:
            loss = (objective_value**2) * objective.direction
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
        self, variables: Dict[str, Variable]
    ) -> Optional[np.ndarray]:
        """
        get combinations of all categorical (binary, enum) variables
        # reuse code in UDAO
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: meshed_cv_value: ndarray, categorical(binary, enum) variables
        """
        cv_value_list = [
            variable.values
            for variable in variables.values()
            if isinstance(variable, EnumVariable)
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
    def within_objective_bounds(obj_value: float, objective: Objective) -> bool:
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

    def get_tensor(self, var: Any, requires_grad: bool = False) -> th.Tensor:
        """
        convert numpy array to tensor
        :param var: ndarray, variable values
        :return: tensor, variable values
        """
        return th.tensor(
            var, device=self.device, dtype=self.dtype, requires_grad=requires_grad
        )
