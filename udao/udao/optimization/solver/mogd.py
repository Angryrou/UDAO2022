import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch as th
import torch.optim as optim
from torch.multiprocessing import Pool

from ...utils.logging import logger
from ..concepts import Constraint, EnumVariable, NumericVariable, Objective, Variable

SEED = 0
DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32
NOT_FOUND_ERROR = "no valid configuration found"


def get_default_device() -> th.device:
    return th.device("cuda") if th.cuda.is_available() else th.device("cpu")


class MOGD:
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
        self.stress = params.stress
        self.seed = params.seed
        self.device = params.device
        self.dtype = params.dtype

    def problem_setup(
        self,
        variables: Sequence[Variable],
        std_func: Optional[Callable],
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        precision_list: list,
        alpha: float = 0.0,
        accurate: bool = True,
    ) -> None:
        """
        set up problem in solver

        :param accurate: bool,
            whether the predictive model is accurate (True) or not (False)
        :param std_func: function,
            passed by the user, for loss calculation
            when the predictive model is not accurate
        :return:
        """
        self.variables = variables
        self.vars_max, self.vars_min = self.get_bounds(self.variables)
        self.categorical_variable_ids = [
            i for i, v in enumerate(variables) if isinstance(v, EnumVariable)
        ]
        self.numerical_variable_ids = [
            i for i, v in enumerate(variables) if isinstance(v, NumericVariable)
        ]
        self.objectives = objectives
        self.constraints = constraints
        self.accurate = accurate
        self.std_func = std_func
        self.alpha = alpha
        self.precision_list = precision_list

    def _single_start_opt(
        self,
        wl_id: str,
        meshed_categorical_vars: np.ndarray,
        batch_size: int,
        obj_bounds_dict: Optional[Dict],
        objective_name: str,
    ) -> Any:
        best_loss = np.inf
        best_objs: Optional[Dict] = None
        best_vars: Optional[np.ndarray] = None
        iter_num = 0
        opt_obj_ind = [
            i for i, obj in enumerate(self.objectives) if obj.name == objective_name
        ][0]
        for bv in meshed_categorical_vars:
            bv_dict = {
                cind: bv[ind] for ind, cind in enumerate(self.categorical_variable_ids)
            }
            numerical_var_list = th.rand(
                batch_size,
                len(self.numerical_variable_ids),
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            optimizer = optim.Adam(
                [numerical_var_list], lr=self.lr, weight_decay=self.wd
            )

            local_best_iter = 0
            local_best_loss = np.inf
            local_best_objs: Optional[Dict] = None
            local_best_var: Optional[np.ndarray] = None
            i = 0

            while i < self.max_iter:
                vars_kernal = self._get_tensor_vars_cat(
                    numerical_var_list,
                    bv_dict,
                )
                vars = vars_kernal.to(self.device)
                if obj_bounds_dict:
                    objs_pred_dict = {
                        cst_obj: self._get_tensor_obj_pred(wl_id, vars, ob_ind).to(
                            self.device
                        )
                        for ob_ind, cst_obj in enumerate(obj_bounds_dict)
                    }
                    loss, loss_id = self._soo_loss(
                        wl_id,
                        vars,
                        objs_pred_dict,
                        obj_bounds_dict,
                        target_obj_name=objective_name,
                        target_obj_ind=opt_obj_ind,
                    )
                else:
                    objs_pred_dict = {
                        objective_name: self._get_tensor_obj_pred(
                            wl_id, vars, opt_obj_ind
                        ).to(self.device)
                    }
                    loss, loss_id = self._unbounded_soo_loss(
                        wl_id, opt_obj_ind, objs_pred_dict, vars
                    )

                if i > 0 and loss.item() < local_best_loss:
                    local_best_loss = loss.item()

                    local_best_objs = {
                        k: v[loss_id].item() for k, v in objs_pred_dict.items()
                    }
                    local_best_var = vars.data.cpu().numpy()[loss_id].copy()
                    # local_best_var = vars.data.numpy().copy()
                    local_best_iter = i

                optimizer.zero_grad()
                loss.backward()  # type: ignore
                optimizer.step()  # update parameters

                constrained_numerical_var_list = (
                    self._get_tensor_numerical_constrained_vars(numerical_var_list.data)
                )
                numerical_var_list.data = constrained_numerical_var_list

                if i > local_best_iter + self.patient:
                    break

                i += 1
            logging.info(
                f"Local best {local_best_objs} at {local_best_iter} with vars:\n"
                f"{local_best_var}"
            )
            if local_best_objs is None or local_best_var is None:
                raise Exception("Unexpected local_best_objs or local_best_var is None.")
            logger.debug(
                f"Finished at iteration {iter}, best local {objective_name} "
                f"found {local_best_objs[objective_name]:.5f}"
                f" \nat iteration {local_best_iter},"
                f" \nwith vars: {self.get_raw_vars(local_best_var)}"
            )

            iter_num += i + 1

            if self.within_constraints(
                wl_id, local_best_var
            ) & self.within_objective_bounds(local_best_objs, obj_bounds_dict):
                if local_best_loss < best_loss:
                    best_objs = local_best_objs
                    best_loss = local_best_loss
                    best_vars = local_best_var

        if self.within_constraints(wl_id, best_vars) & self.within_objective_bounds(
            best_objs, obj_bounds_dict
        ):
            if best_vars is None:
                raise Exception("Unexpected best_vars is None.")
            best_raw_vars = self.get_raw_vars(best_vars)
            obj_pred_dict = self._get_obj_pred_dict(wl_id, best_raw_vars)
            target_obj_val = obj_pred_dict[objective_name]
            logger.debug(
                f"get best {objective_name}: {best_objs} at {best_vars}"
                f" with {iter_num} iterations, loss = {best_loss}"
            )
        else:
            obj_pred_dict = None
            best_raw_vars = None
            target_obj_val = np.inf
            logger.debug("No valid solutions and variables found!")
        return obj_pred_dict, best_raw_vars, best_loss, target_obj_val

    def optimize_constrained_so(
        self,
        wl_id: str,
        objective_name: str,
        obj_bounds_dict: Optional[Dict],
        batch_size: int = 1,
    ) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
        """
        solve single objective optimization constrained by objective values
        :param wl_id: str, workload id, e.g. '1-7'
        :param objective_name: str, name of objective to be optimized
        :param obj_bounds_dict: dict, keys are the name of objectives,
            values are the lower and upper var_ranges for each objective value
        :return:
                objs: list, all objective values
                vars: list, variable values
        """

        th.manual_seed(self.seed)
        if not self._check_obj(objective_name):
            raise Exception(
                f"Objective {objective_name}" "was not part of the problem definition."
            )
        if obj_bounds_dict is not None:
            for cst_obj in obj_bounds_dict:
                if not self._check_obj(cst_obj):
                    raise ValueError(
                        f"Objective {cst_obj} was not part of the problem definition."
                    )
        meshed_categorical_vars = self.get_meshed_categorical_vars(self.variables)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = np.array([0])

        best_loss_list: List[float] = []
        objs_list = []
        vars_list = []
        target_obj_values = []
        for _ in range(self.multistart):
            (
                obj_pred_dict,
                best_raw_vars,
                best_loss,
                target_obj_val,
            ) = self._single_start_opt(
                meshed_categorical_vars=meshed_categorical_vars,
                batch_size=batch_size,
                obj_bounds_dict=obj_bounds_dict,
                wl_id=wl_id,
                objective_name=objective_name,
            )
            best_loss_list.append(best_loss)
            objs_list.append(obj_pred_dict)
            vars_list.append(best_raw_vars)
            target_obj_values.append(target_obj_val)
        idx = np.argmin(target_obj_values)
        vars_cand = vars_list[idx]
        if vars_cand is not None:
            obj_cand = objs_list[idx]
            if obj_cand is None:
                raise Exception(f"Unexpected objs_list[{idx}] is None.")
            objs = list(obj_cand.values())
            return_vars = vars_cand.reshape([len(self.variables)])
        else:
            objs, return_vars = None, None

        return objs, return_vars

    def optimize_constrained_so_parallel(
        self,
        wl_id: str,
        objective_name: str,
        cell_list: list,
        batch_size: int = 1,
    ) -> List[Tuple[Optional[List[float]], Optional[np.ndarray]]]:
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
            ret_list = pool.starmap(self.optimize_constrained_so, arg_list)
        return ret_list

    ##################
    ## _loss        ##
    ##################
    def _constraints_loss(self, wl_id: str, vars: th.Tensor) -> th.Tensor:
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
        if vars.ndim == 1:
            vars = vars.reshape([1, vars.shape[0]])
        const_violation = self.get_tensor(0)
        for i, constraint in enumerate(self.constraints):
            constraint_func = constraint.function
            if isinstance(constraint_func, th.nn.Module):
                constraint_func = constraint_func.to(self.device)
            if constraint.type == "<=":
                const_violation = th.relu(
                    const_violation + constraint.function(vars, wl_id)
                )
            elif constraint.type == "==":
                const_violation1 = th.relu(
                    const_violation + constraint.function(vars, wl_id)
                )
                const_violation2 = th.relu(
                    const_violation + (constraint.function(vars, wl_id)) * (-1)
                )
                const_violation = const_violation1 + const_violation2
            elif constraint.type == ">=":
                const_violation = th.relu(
                    const_violation + (constraint.function(vars, wl_id)) * (-1)
                )
            else:
                raise Exception(f"{constraint.type} is not supported!")

        if const_violation.sum() != 0:
            const_loss = const_violation**2 + 1e5
        else:
            const_loss = self.get_tensor(0)

        return const_loss

    def _unbounded_soo_loss(
        self, wl_id: str, obj_ind: int, pred_dict: Dict, vars: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        compute loss fixme: double-check for the objective
        with negative values (e.g. throughput)
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj_ind: int, the index of the objective to be optimized
        :param obj_pred: tensor, objective value(prediction)
        :param vars: tensor ((bs, n_vars) or (n_vars, )),
        variables, where bs is batch_size
        :return: [tensor, tensor], minimum loss and its index
        """
        obj_pred = pred_dict[self.objectives[obj_ind].name]
        if not self.accurate:
            if self.std_func is None:
                raise ValueError(
                    "std_func must be provided if accurate is set to False"
                )
            std = self.std_func(wl_id, vars, self.objectives[obj_ind].name)
            loss = ((obj_pred + std * self.alpha) ** 2) * self.objectives[
                obj_ind
            ].direction
        else:
            loss = (obj_pred**2) * self.objectives[obj_ind].direction
        loss = loss.to(self.device)
        const_loss = self._constraints_loss(wl_id, vars).to(self.device)
        loss = loss + const_loss
        return th.min(loss), th.argmin(loss)

    def _soo_loss(
        self,
        wl_id: str,
        vars: th.Tensor,
        pred_dict: Dict,
        obj_bounds: Dict,
        target_obj_name: str,
        target_obj_ind: int,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        compute loss constrained by objective values fixme:
        double-check for the objective with negative values (e.g. throughput)
        # reuse code in UDAO
        :param wl_id: str, workload id, e.g. '1-7'
        :param vars: tensor ((bs, n_vars) or (n_vars, )), variables,
            where bs is batch_size
        :param pred_dict: dict, keys are objective names,
            values are objective values
        :param obj_bounds: dict, keys are objective names, values
        are lower and upper var_ranges of each objective value
        :param target_obj_name: str, the name of the objective to be optimized
        :param target_obj_ind: int, the index of target_obj_name
        :return:
                loss: tensor (Tensor())
        """
        loss_shape = (1) if vars.ndim == 1 else (vars.shape[0])
        loss = th.zeros(loss_shape, device=self.device, dtype=self.dtype)

        for cst_obj, [lower, upper] in obj_bounds.items():
            lower = lower.to(self.device)
            upper = upper.to(self.device)
            # assert pred_dict[cst_obj].shape[0] == 1
            obj_pred_raw = pred_dict[cst_obj].sum(-1)  # (1,1)
            if not self.accurate:
                if self.std_func is None:
                    raise ValueError(
                        "std_func must be provided if accurate is set to False"
                    )
                std = self.std_func(wl_id, vars, cst_obj).to(self.device)  # type: ignore
                assert std.shape[0] == 1 and std.shape[1] == 1
                obj_pred = obj_pred_raw + std.sum(-1) * self.alpha
            else:
                obj_pred = obj_pred_raw

            if upper != lower:
                norm_cst_obj_pred = (obj_pred - lower) / (upper - lower)  # scaled
                add_loss = th.where(
                    (norm_cst_obj_pred < 0) | (norm_cst_obj_pred > 1),
                    (norm_cst_obj_pred - 0.5) ** 2 + self.stress,
                    norm_cst_obj_pred * self.objectives[target_obj_ind].direction
                    if cst_obj == target_obj_name
                    else 0,
                )

            else:
                add_loss = (obj_pred - upper) ** 2 + self.stress
            loss = loss + add_loss.to(self.device)
        loss = loss + self._constraints_loss(wl_id, vars).to(self.device)
        return th.min(loss), th.argmin(loss)

    ##################
    ## _get (vars)  ##
    ##################

    def get_meshed_categorical_vars(
        self, variables: Sequence[Variable]
    ) -> Optional[np.ndarray]:
        """
        get combinations of all categorical (binary, enum) variables
        # reuse code in UDAO
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: meshed_cv_value: ndarray, categorical(binary, enum) variables
        """

        cv_value_list = [
            variable.values
            for variable in variables
            if isinstance(variable, EnumVariable)
        ]
        if not cv_value_list:
            return None
        meshed_cv_value_list = [x_.reshape(-1, 1) for x_ in np.meshgrid(*cv_value_list)]
        meshed_cv_value = np.concatenate(meshed_cv_value_list, axis=1)
        return meshed_cv_value

    def _get_tensor_vars_cat(
        self,
        numerical_var_list: th.Tensor,
        cv_dict: Dict,
    ) -> th.Tensor:
        """
        concatenate values of numerical(float, integer)
        and categorical(binary, enum) variables together
        (reuse code in UDAO)
        :param numerical_var_list: tensor((bs, len(numerical_var_inds))
            or (numerical_var_inds,)), values of numercial variables
        :param cv_dict: dict(key: indices of (binary, enum) variables,
            value: value of (binary, enum) variables),
            indices and values of bianry variables
        :return: vars: tensor((bs, len(numerical_var_inds))
            or (numerical_var_inds,)), values of all variables
        """

        to_concat = []
        ck_ind = 0
        if numerical_var_list.ndimension() == 1:
            for i, variable in enumerate(self.variables):
                if isinstance(variable, NumericVariable):
                    to_concat.append(numerical_var_list[i - ck_ind : i + 1 - ck_ind])
                elif isinstance(variable, EnumVariable):
                    to_concat.append(self.get_tensor([cv_dict[i]]))
                    ck_ind += 1
                else:
                    raise Exception(f"unsupported type in var {i}")
            vars = th.cat(to_concat)
        else:
            n_batch = numerical_var_list.shape[0]
            for i, variable in enumerate(self.variables):
                if isinstance(variable, NumericVariable):
                    to_concat.append(numerical_var_list[:, i - ck_ind : i + 1 - ck_ind])
                elif isinstance(variable, EnumVariable):
                    to_concat.append(
                        th.ones((n_batch, 1), device=self.device, dtype=self.dtype)
                        * cv_dict[i]
                    )
                    ck_ind += 1
                else:
                    raise Exception(f"unsupported type in var {i}")
            vars = th.cat(to_concat, dim=1)
            assert vars.shape[0] == numerical_var_list.shape[0] and vars.shape[
                1
            ] == len(self.variables)
        return vars

    def _get_tensor_numerical_constrained_vars(
        self,
        numerical_vars: th.Tensor,
    ) -> th.Tensor:
        """
        make the values of numerical variables within their range
        :param numerical_var_list: tensor ((bs, len(numerical_var_inds)
            or (len(numerical_var_ids), ), values of numerical variables
            (FLOAT and INTEGER)
        :return:
               Tensor(n_numerical_inds,),
                    normalized numerical variables
        """

        bounded_np = th.clip(numerical_vars, 0, 1).cpu().numpy()

        # adjust variable values to its pickable points
        raw_np = self.get_raw_vars(
            bounded_np,
            normalized_ids=self.numerical_variable_ids,
        )
        normalized_np = self.get_normalized_vars(
            raw_np, normalized_ids=self.numerical_variable_ids
        )
        return self.get_tensor(normalized_np)

    # reuse code in UDAO
    def get_raw_vars(
        self,
        normalized_vars: np.ndarray,
        normalized_ids: Optional[list] = None,
    ) -> np.ndarray:
        """
        denormalize the values of each variable
        :param normalized_vars: ndarray((bs, n_vars) or (n_vars,)), normalized variables
        :param normalized_ids: list, indices of numerical variables (float and integer)
        :return: raw_vars: ndarray((bs, n_vars) or (n_vars,)), denormalized variables
        """
        vars_max = (
            self.vars_max if normalized_ids is None else self.vars_max[normalized_ids]
        )
        vars_min = (
            self.vars_min if normalized_ids is None else self.vars_min[normalized_ids]
        )
        precision_list = (
            self.precision_list
            if normalized_ids is None
            else np.array(self.precision_list)[normalized_ids].tolist()
        )
        vars = normalized_vars * (vars_max - vars_min) + vars_min
        raw_vars = np.array(
            [c.round(p) for c, p in zip(vars.astype(float).T, precision_list)]
        ).T
        return raw_vars

    # reuse code in UDAO
    def get_normalized_vars(
        self,
        raw_vars: np.ndarray,
        normalized_ids: Optional[list] = None,
    ) -> np.ndarray:
        """
        normalize the values of each variable
        :param raw_vars: ndarray((bs, n_vars) or (n_vars, )),
            raw variable values (bounded with original lower and upper var_ranges)
        :param normalized_ids: list, indices fo numerical variables (float and integer)
        :return: normalized_vars, ndarray((bs, n_vars) or (n_vars, )),
            normalized variable values
        """
        vars_max = (
            self.vars_max if normalized_ids is None else self.vars_max[normalized_ids]
        )
        vars_min = (
            self.vars_min if normalized_ids is None else self.vars_min[normalized_ids]
        )
        normalized_vars = (raw_vars - vars_min) / (vars_max - vars_min)
        return normalized_vars

    def get_bounds(
        self, variables: Sequence[Variable]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        get min and max values for each variable
        :param var_ranges: ndarray (n_vars,),
            the lower and upper var_ranges of non-ENUM variables,
            and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: maximum and minimum values of all variables
                ndarray(n_vars,): the maximum values of all variables
                ndarray(n_vars,): the minimum values of all variables
        """
        var_max, var_min = [], []
        for variable in variables:
            if isinstance(variable, NumericVariable):
                var_max.append(variable.upper)
                var_min.append(variable.lower)
            elif isinstance(variable, EnumVariable):
                var_max.append(max(variable.values))
                var_min.append(min(variable.values))
            else:
                Exception(f"Variable type {type(variable)} is not supported!")

        return np.array(var_max), np.array(var_min)

    ##################
    ## _get (objs)  ##
    ##################
    def _get_tensor_obj_pred(
        self, wl_id: str, vars: Union[np.ndarray, th.Tensor], obj_ind: int
    ) -> th.Tensor:
        """
        get objective values
        :param wl_id: str, workload id, e.g. '1-7'
        :param vars: tensor ((bs, n_vars) or (n_vars, )),
            variables, where bs is batch_size
        :param obj_ind: int, the index of objective to optimize
        :return: obj_pred: tensor(1,1), the objective value
        """
        obj_function = self.objectives[obj_ind].function
        isinstance(obj_function, th.nn.Module)
        if isinstance(obj_function, th.nn.Module):
            obj_function = obj_function.to(self.device)
        if not th.is_tensor(vars):  # type: ignore
            vars = self.get_tensor(vars)
        vars = vars.to(self.device)  # type: ignore
        if vars.ndim == 1:
            obj_pred = cast(
                th.Tensor,
                obj_function(vars.reshape([1, vars.shape[0]]), wl_id),
            )
        else:
            obj_pred = cast(th.Tensor, obj_function(vars, wl_id))

        assert obj_pred.ndimension() == 2
        return obj_pred

    def _get_obj_pred_dict(self, wl_id: str, best_raw_vars: np.ndarray) -> Dict:
        """
        get objective values
        :param wl_id: str, workload id, e.g. '1-7'
        :param best_obj_dict: dict, keys are objective names,
            values are objective values,
            e.g. {'latency': 12406.1416015625, 'cores': 48.0}
        :param best_raw_vars: ndarray(n_vars,), variable values
        :return: obj_pred_dict: dict, keys are objective names,
            values are objective values
        """
        vars_norm = self.get_normalized_vars(best_raw_vars)
        obj_pred_dict = {
            objective.name: self._get_tensor_obj_pred(wl_id, vars_norm, obj_i)[
                0, 0
            ].item()
            for obj_i, objective in enumerate(self.objectives)
        }
        return obj_pred_dict

    ##################
    ## _check       ##
    ##################

    def within_variable_range(self, vars_raw: np.ndarray) -> bool:
        """
        check whether the variable are available
        (within the range setting of variables in the config.json file)
        :param vars_raw: ndarray (n_vars,), variable values
        :param var_types: list, variable types
            (float, integer, binary, enum)
        :param var_ranges: ndarray (n_vars,), the lower and upper
            var_ranges of non-ENUM variables, and values of ENUM variables
        :return: bool, whether available (True)
        """
        vars_max, vars_min = self.get_bounds(self.variables)
        return (
            True
            if np.all(vars_max - vars_raw >= 0) and np.all(vars_min - vars_raw <= 0)
            else False
        )

    def _check_obj(self, obj_name: str) -> bool:
        """
        check whether the objectives are avaiable
        :param obj: str, objective names
        :return: bool, whether available (True)
        """
        return any(objective.name == obj_name for objective in self.objectives)

    # check violations of constraint functions
    def within_constraints(self, wl_id: str, var_array: Optional[np.ndarray]) -> bool:
        """
        check whether the best variable values resulting
        in violation of constraint functions
        :param wl_id: str, workload id, e.g. '1-7'
        :param best_var: ndarray(n_vars, ), best variable values for each variable
        :return: bool, whether it returns feasible solutions (True) or not (False)
        """
        if var_array is None:
            return False

        var_tensor = self.get_tensor(var_array)

        if var_tensor.ndim == 1:
            var_tensor.reshape([1, var_array.shape[0]])
        const_loss = self._constraints_loss(wl_id, var_tensor)
        if const_loss <= 0:
            return True
        else:
            return False

    # check violations of objective value var_ranges
    # reuse code in UDAO
    def within_objective_bounds(
        self, pred_dict: Optional[Dict], obj_bounds: Optional[Dict]
    ) -> bool:
        """
        check whether violating the objective value var_ranges
        :param pred_dict: dict, keys are objective names,
        values are objective values
        :param obj_bounds: dict, keys are objective names,
        values are lower and upper var_ranges of each objective value
        :return: True or False
        """
        if obj_bounds is None:
            return True
        if pred_dict is None:
            return False
        for obj, obj_pred in pred_dict.items():
            lower, upper = obj_bounds[obj]
            if lower.item() <= obj_pred <= upper.item():
                pass
            else:
                return False
        return True

    def get_tensor(self, var: Any, requires_grad: bool = False) -> th.Tensor:
        """
        convert numpy array to tensor
        :param var: ndarray, variable values
        :return: tensor, variable values
        """
        return th.tensor(
            var, device=self.device, dtype=self.dtype, requires_grad=requires_grad
        )
