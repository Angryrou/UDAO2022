import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch as th
import torch.optim as optim
from torch.multiprocessing import Pool

from ..concepts import Constraint, EnumVariable, NumericVariable, Objective, Variable
from ..utils import solver_utils as solver_ut

SEED = 0
DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32
NOT_FOUND_ERROR = "no valid configuration found"
CHECK_FALSE_RET = "-1"


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

        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        self.dtype = th.float32

    def _problem(
        self,
        variables: List[Variable],
        std_func: Optional[Callable],
        objectives: List[Objective],
        constraints: List[Constraint],
        alpha: float = 0.0,
        accurate: bool = True,
    ) -> None:
        """
        set up problem in solver
        :param wl_list: list, each element is a string,
            representing workload id (wl_id), e.g. '1-7'
        :param wl_ranges: function,
            provided by users, to return upper and lower bounds of variables
        :param vars_constraints: dict,
            key is 'vars_min' and 'vars_max', value is min and max values
            for all variables of one workload. The purpose is to put
            variable values into a region where the model performs better.
        :param accurate: bool,
            whether the predictive model is accurate (True) or not (False)
        :param std_func: function,
            passed by the user, for loss calculation
            when the predictive model is not accurate
        :param obj_funcs: list, objective functions
        :param obj_names: list, objective names
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types
            ("<=" "==" or "<", e.g. g1(x1, x2, ...) - c <= 0)
        :return:
        """
        self.variables = variables
        self.objectives = objectives
        self.constraints = constraints
        self.accurate = accurate
        self.std_func = std_func
        self.alpha = alpha

    def single_objective_opt(
        self,
        wl_id: str,
        obj: str,
        opt_obj_ind: int,
        precision_list: list,
        bs: int = 16,
        verbose: bool = False,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        solve (constrained) single-objective optimization
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj: str, objective name
        :param accurate: bool, whether the predictive model
            is accurate (True) or not (False)
        :param alpha: float, the value used in loss calculation of
            the inaccurate model
        :param opt_obj_ind: int, the index of objective to optimize
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_ranges: ndarray (n_vars,), the lower and upper
            var_ranges of non-ENUM variables, and values of ENUM variables
        :param precision_list: list, precision of each variable
        :param bs: int, batch size
        :param verbose: bool, to print further information if needed
        :return:
                objs_list[idx]: float, objective value
                vars: ndarray(1, n_vars), variable values
        """

        th.manual_seed(self.seed)
        vars_max, vars_min = self.get_bounds(self.variables)
        if not self._check_obj(obj):
            raise Exception(f"Workload {wl_id} or objective {obj}" "is not supported!")
        assert self.multistart >= 1

        meshed_categorical_vars = self.get_meshed_categorical_vars(self.variables)
        numerical_var_inds = self.get_numerical_var_inds(self.variables)
        categorical_var_inds = self.get_categorical_var_inds(self.variables)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = np.array([0])

        best_loss_list: List[float] = []
        objs_list: List[float] = []
        vars_list: List[Optional[np.ndarray]] = []
        for _ in range(self.multistart):
            best_loss, best_obj, best_vars, iter_num = np.inf, np.inf, None, 0
            for bv in meshed_categorical_vars:
                bv_dict = {
                    cind: bv[ind] for ind, cind in enumerate(categorical_var_inds)
                }
                numerical_var_list = th.rand(
                    bs,
                    len(numerical_var_inds),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True,
                )
                optimizer = optim.Adam(
                    [numerical_var_list], lr=self.lr, weight_decay=self.wd
                )

                local_best_iter = 0
                local_best_loss = np.inf
                local_best_obj: Optional[float] = None
                local_best_var: Optional[np.ndarray] = None
                i = 0
                while i < self.max_iter:
                    vars_kernal = self._get_tensor_vars_cat(
                        numerical_var_inds,
                        numerical_var_list,
                        categorical_var_inds,
                        bv_dict,
                    )
                    vars = vars_kernal
                    obj_pred = self._get_tensor_obj_pred(
                        wl_id, vars, opt_obj_ind
                    )  # Nx1
                    loss, loss_id = self._loss_soo_minibatch(
                        wl_id, opt_obj_ind, obj_pred, vars
                    )

                    if i > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()

                        local_best_obj = obj_pred[loss_id].item()
                        if vars.ndim == 1:
                            local_best_var = vars.data.numpy()[loss_id].copy()
                        else:
                            local_best_var = vars.data.numpy()[loss_id, :].copy()
                        local_best_iter = i

                    optimizer.zero_grad()
                    loss.backward()  # type: ignore
                    optimizer.step()  # update parameters

                    constrained_numerical_var_list = (
                        self._get_tensor_numerical_constrained_vars(
                            numerical_var_list.data,
                            numerical_var_inds,
                            precision_list,
                        )
                    )
                    numerical_var_list.data = constrained_numerical_var_list

                    if i > local_best_iter + self.patient:
                        # early stop
                        break

                    if verbose:
                        if i % 10 == 0:
                            print(
                                f"iteration {i}, {obj}: {obj_pred[loss_id].item():.2f}"
                            )
                            print(vars)
                    i += 1
                logging.info(
                    f"Local best {obj}: {local_best_obj:.5f} "
                    f"at {local_best_iter} with vars:\n"
                    f"{local_best_var}"
                )
                if local_best_var is None:
                    raise Exception("Unexpected local_best_var is None.")
                if verbose:
                    denormalized = self.get_raw_vars(
                        local_best_var, vars_max, vars_min, precision_list
                    )
                    print(
                        f"Finished at iteration {i}, best local "
                        f"{obj} found as {local_best_obj:.5f}"
                        f" \nat iteration {local_best_iter},"
                        f" \nwith vars: {denormalized}"
                    )

                iter_num += i + 1

                if self.check_const_func_vio(wl_id, local_best_var):
                    if local_best_obj is None:
                        raise Exception("Unexpected local_best_obj is None.")
                    if local_best_loss < best_loss:
                        best_obj = local_best_obj
                        best_loss = local_best_loss
                        best_vars = local_best_var

            if self.check_const_func_vio(wl_id, best_vars):
                if best_vars is None:
                    raise Exception("Unexpected best_vars is None.")
                best_raw_vars = self.get_raw_vars(
                    best_vars, vars_max, vars_min, precision_list
                )
                logging.info(
                    f"get best {obj}: {best_obj} at {best_raw_vars} "
                    f"with {iter_num} iterations, loss = {best_loss}"
                )

                if verbose:
                    print()
                    print("*" * 10)
                    print(
                        f"get best {obj}: {best_obj} at {best_raw_vars} "
                        f"with {iter_num} iterations, loss = {best_loss}"
                    )
            else:
                best_raw_vars = None
                if verbose:
                    print("No valid solutions and variables found!")

            best_loss_list.append(best_loss)
            objs_list.append(best_obj)
            vars_list.append(best_raw_vars)

        idx = np.argmin(best_loss_list)
        vars_cand = vars_list[idx]

        if vars_cand is not None:
            return_vars = vars_cand.reshape([1, len(self.variables)])
        else:
            return_vars = None
        return objs_list[idx], return_vars

    def constraint_so_opt(
        self,
        wl_id: str,
        obj: str,
        opt_obj_ind: int,
        obj_bounds_dict: Dict,
        precision_list: list,
        verbose: bool = False,
    ) -> Tuple[List[float] | None, np.ndarray | None]:
        """
        solve single objective optimization constrained by objective values
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj: str, name of objective to be optimized
        :param accurate: bool,
            whether the predictive model is accurate (True) or not (False)
        :param alpha: float,
            the value used in loss calculation of the inaccurate model
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types
        :param var_range: ndarray (n_vars,), the lower and upper
            var_ranges of non-ENUM variables, and values of ENUM variables
        :param obj_bounds_dict: dict, keys are the name of objectives,
            values are the lower and upper var_ranges for each objective value
        :param precision_list: list, precision of each variable
        :param verbose: bool, to print further information if needed
        :param is_parallel: bool,
            whether it is called parallelly (True) or not (False)
        :return:
                objs: list, all objective values
                vars: ndarray(1, n_vars), variable values
        """

        th.manual_seed(self.seed)
        if not self._check_obj(obj):
            raise Exception(
                f"Workload {wl_id} or objective {obj}"
                "was not part of the problem definition."
            )
        for cst_obj in obj_bounds_dict:
            if not self._check_obj(cst_obj):
                raise Exception(
                    f"Objective {cst_obj} was not part of the problem definition."
                )
        vars_max, vars_min = self.get_bounds(self.variables)
        meshed_categorical_vars = self.get_meshed_categorical_vars(self.variables)
        numerical_var_inds = self.get_numerical_var_inds(self.variables)
        categorical_var_inds = self.get_categorical_var_inds(self.variables)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = np.array([0])

        best_loss_list = []
        objs_list, vars_list = [], []
        target_obj_val = []
        for _ in range(self.multistart):
            best_loss, best_objs, best_vars, iter_num = np.inf, None, None, 0

            for bv in meshed_categorical_vars:
                bv_dict = {
                    cind: bv[ind] for ind, cind in enumerate(categorical_var_inds)
                }
                numerical_var_list = th.rand(
                    len(numerical_var_inds),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True,
                )
                optimizer = optim.Adam(
                    [numerical_var_list], lr=self.lr, weight_decay=self.wd
                )

                local_best_iter = 0
                local_best_loss = np.inf
                local_best_objs: Dict | None = None
                local_best_var: np.ndarray | None = None
                i = 0

                while i < self.max_iter:
                    vars_kernal = self._get_tensor_vars_cat(
                        numerical_var_inds,
                        numerical_var_list,
                        categorical_var_inds,
                        bv_dict,
                    )
                    vars = vars_kernal
                    print(vars)
                    objs_pred_dict = {
                        cst_obj: self._get_tensor_obj_pred(wl_id, vars, ob_ind)
                        for ob_ind, cst_obj in enumerate(obj_bounds_dict)
                    }
                    loss = self._loss_soo(
                        wl_id,
                        vars,
                        objs_pred_dict,
                        obj_bounds_dict,
                        target_obj_name=obj,
                        target_obj_ind=opt_obj_ind,
                    )

                    if i > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_objs = {
                            k: v.item() for k, v in objs_pred_dict.items()
                        }
                        local_best_var = vars.data.numpy().copy()
                        local_best_iter = i

                    optimizer.zero_grad()
                    loss.backward()  # type: ignore
                    optimizer.step()  # update parameters

                    constrained_numerical_var_list = (
                        self._get_tensor_numerical_constrained_vars(
                            numerical_var_list.data,
                            numerical_var_inds,
                            precision_list,
                        )
                    )
                    numerical_var_list.data = constrained_numerical_var_list

                    if i > local_best_iter + self.patient:
                        # early stop
                        break

                    if verbose:
                        if i % 10 == 0:
                            print(
                                f"iteration {iter}, {obj}: "
                                f"{objs_pred_dict[obj].item():.2f}"
                            )
                            print(vars)
                    i += 1
                logging.info(
                    f"Local best {local_best_objs} at {local_best_iter} with vars:\n"
                    f"{local_best_var}"
                )
                if local_best_objs is None or local_best_var is None:
                    raise Exception(
                        "Unexpected local_best_objs or local_best_var is None."
                    )
                if verbose:
                    displayed_vars = self.get_raw_vars(
                        local_best_var, vars_max, vars_min, precision_list
                    )
                    print(
                        f"Finished at iteration {iter}, "
                        f"best local {obj} found as {local_best_objs[obj]:.5f}"
                        f" \nat iteration {local_best_iter},"
                        f" \nwith vars: {displayed_vars}"
                    )

                iter_num += i + 1

                if self.check_const_func_vio(
                    wl_id, local_best_var
                ) & self.check_obj_bounds_vio(local_best_objs, obj_bounds_dict):
                    if local_best_loss < best_loss:
                        best_objs = local_best_objs
                        best_loss = local_best_loss
                        best_vars = local_best_var

            if self.check_const_func_vio(wl_id, best_vars) & self.check_obj_bounds_vio(
                best_objs, obj_bounds_dict
            ):
                if best_vars is None:
                    raise Exception("Unexpected best_vars is None.")
                best_raw_vars = self.get_raw_vars(
                    best_vars, vars_max, vars_min, precision_list
                )
                obj_pred_dict = self._get_obj_pred_dict(
                    wl_id, obj_bounds_dict, best_raw_vars
                )
                target_obj_val.append(obj_pred_dict[obj])
                if verbose:
                    print()
                    print("*" * 10)
                    print(
                        f"get best {obj}: {best_objs} at {best_vars}"
                        f" with {iter_num} iterations, loss = {best_loss}"
                    )
            else:
                obj_pred_dict = None
                best_raw_vars = None
                target_obj_val.append(np.inf)
                if verbose:
                    print("No valid solutions and variables found!")

            best_loss_list.append(best_loss)
            objs_list.append(obj_pred_dict)
            vars_list.append(best_raw_vars)

        idx = np.argmin(target_obj_val)
        vars_cand = vars_list[idx]
        if vars_cand is not None:
            obj_cand = objs_list[idx]
            if obj_cand is None:
                raise Exception(f"Unexpected objs_list[{idx}] is None.")
            objs = list(obj_cand.values())
            return_vars = vars_cand.reshape([1, len(self.variables)])
        else:
            objs, return_vars = None, None

        return objs, return_vars

    def constraint_so_parallel(
        self,
        wl_id: str,
        obj: str,
        opt_obj_ind: int,
        precision_list: list,
        cell_list: list,
    ) -> List[Tuple[Optional[List[float]], Optional[np.ndarray]]]:
        """
        solve the single objective optimization
        constrained by objective values parallelly
        :param wl_id: str, workload id, e.g. '1-7'
        :param obj: str, name of objective to be optimized
        :param accurate: bool,
            whether the predictive model is accurate (True) or not (False)
        :param alpha: float,
            the value used in loss calculation of the inaccurate model
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types
        :param var_ranges: ndarray (n_vars,), the lower and upper
            var_ranges of non-ENUM variables, and values of ENUM variables
        :param precision_list: list, precision of each variable
        :param cell_list: list, each element is a dict to indicate
            the var_ranges of objective values
        :param verbose: bool, to print further information if needed
        :return:
                ret_list: list,
                    each element is a solution
                    tuple with size 2) with objective values (tuple[0], list)
                    and variables (tuple[1], ndarray(1, n_vars))
        """

        vars_max, vars_min = self.get_bounds(self.variables)

        th.manual_seed(self.seed)

        # generate the list of input parameters for constraint_so_opt
        arg_list = [
            (
                wl_id,
                obj,
                opt_obj_ind,
                obj_bounds_dict,
                precision_list,
                False,
            )
            for obj_bounds_dict in cell_list
        ]

        # call self.constraint_so_opt parallely
        with Pool(processes=self.process) as pool:
            ret_list = pool.starmap(self.constraint_so_opt, arg_list)

        return ret_list

    ##################
    ## _loss        ##
    ##################
    def _get_tensor_loss_const_funcs(self, wl_id: str, vars: th.Tensor) -> th.Tensor:
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
        const_violation = th.tensor(0, device=self.device, dtype=self.dtype)
        for i, constraint in enumerate(self.constraints):
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
            const_loss = th.tensor(0, device=self.device, dtype=self.dtype)

        return const_loss

    def _loss_soo_minibatch(
        self, wl_id: str, obj_ind: int, obj_pred: th.Tensor, vars: th.Tensor
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
        const_loss = self._get_tensor_loss_const_funcs(wl_id, vars)
        loss = loss + const_loss
        return th.min(loss), th.argmin(loss)

    def _loss_soo(
        self,
        wl_id: str,
        vars: th.Tensor,
        pred_dict: Dict,
        obj_bounds: Dict,
        target_obj_name: str,
        target_obj_ind: int,
    ) -> th.Tensor:
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
        loss = th.tensor(0, device=self.device, dtype=self.dtype)

        for cst_obj, [lower, upper] in obj_bounds.items():
            assert pred_dict[cst_obj].shape[0] == 1
            obj_pred_raw = pred_dict[cst_obj].sum()  # (1,1)
            if not self.accurate:
                if self.std_func is None:
                    raise ValueError(
                        "std_func must be provided if accurate is set to False"
                    )
                std = self.std_func(wl_id, vars, cst_obj)
                assert std.shape[0] == 1 and std.shape[1] == 1
                obj_pred = obj_pred_raw + std.sum() * self.alpha
            else:
                obj_pred = obj_pred_raw

            if upper != lower:
                norm_cst_obj_pred = (obj_pred - lower) / (upper - lower)  # scaled
                add_loss = th.tensor(0, device=self.device, dtype=self.dtype)
                if cst_obj == target_obj_name:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss += (
                            norm_cst_obj_pred
                            * self.objectives[target_obj_ind].direction
                        )
                else:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
            else:
                add_loss = (obj_pred - upper) ** 2 + self.stress
            loss = loss + add_loss
        loss = loss + self._get_tensor_loss_const_funcs(wl_id, vars)
        return loss

    ##################
    ## _get (vars)  ##
    ##################

    def get_meshed_categorical_vars(
        self, variables: List[Variable]
    ) -> np.ndarray | None:
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

    def get_categorical_var_inds(self, variables: List[Variable]) -> List[int]:
        """
        get indices of categorical (binary, enum) variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: categorical_var_inds: list, indices of
            categorical (binary, enum) variables
        """
        categorical_var_inds = [
            i
            for i, variable in enumerate(variables)
            if isinstance(variable, EnumVariable)
        ]
        return categorical_var_inds

    def get_numerical_var_inds(self, variables: List[Variable]) -> list:
        """
        get indices of numerical (float, integer) variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: numerical_var_inds: list, indices of
            numerical (float, integer) variables
        """
        numerical_var_inds = [
            ind for ind, v_ in enumerate(variables) if isinstance(v_, NumericVariable)
        ]
        return numerical_var_inds

    def _get_tensor_vars_cat(
        self,
        numerical_var_inds: list,
        numerical_var_list: th.Tensor,
        categorical_var_inds: list,
        cv_dict: Dict,
    ) -> th.Tensor:
        """
        concatenate values of numerical(float, integer)
        and categorical(binary, enum) variables together
        (reuse code in UDAO)
        :param numerical_var_inds: list, indices of numerical variables
        :param numerical_var_list: tensor((bs, len(numerical_var_inds))
            or (numerical_var_inds,)), values of numercial variables
        :param categorical_var_inds: list, indices of (binary, enum) variables
        :param cv_dict: dict(key: indices of (binary, enum) variables,
            value: value of (binary, enum) variables),
            indices and values of bianry variables
        :return: vars: tensor((bs, len(numerical_var_inds))
            or (numerical_var_inds,)), values of all variables
        """
        #
        # vars is the variables
        target_len = len(numerical_var_inds) + len(categorical_var_inds)
        to_concat = []
        ck_ind = 0
        if numerical_var_list.ndimension() == 1:
            for i in range(target_len):
                if i in set(numerical_var_inds):
                    to_concat.append(numerical_var_list[i - ck_ind : i + 1 - ck_ind])
                elif i in set(categorical_var_inds):
                    to_concat.append(solver_ut.get_tensor([cv_dict[i]]))
                    ck_ind += 1
                else:
                    raise Exception(f"unsupported type in var {i}")
            vars = th.cat(to_concat)
        else:
            n_batch = numerical_var_list.shape[0]
            for i in range(target_len):
                if i in set(numerical_var_inds):
                    to_concat.append(numerical_var_list[:, i - ck_ind : i + 1 - ck_ind])
                elif i in set(categorical_var_inds):
                    to_concat.append(
                        th.ones((n_batch, 1), device=self.device, dtype=self.dtype)
                        * cv_dict[i]
                    )
                    ck_ind += 1
                else:
                    raise Exception(f"unsupported type in var {i}")
            vars = th.cat(to_concat, dim=1)
            assert (
                vars.shape[0] == numerical_var_list.shape[0]
                and vars.shape[1] == target_len
            )
        return vars

    def _get_tensor_numerical_constrained_vars(
        self,
        numerical_var_list: th.Tensor,
        numerical_var_inds: list,
        precision_list: list,
    ) -> th.Tensor:
        """
        make the values of numerical variables within their range
        :param numerical_var_list: tensor ((bs, len(numerical_var_inds)
            or (len(numerical_var_ids), ), values of numerical variables
            (FLOAT and INTEGER)
        :param numerical_var_inds: list, indices of numerical variables
            (float and integer)
        :param vars_max: ndarray(n_vars, ), upper var_ranges of each variable
        :param vars_min: ndarray(n_vars, ), lower var_ranges of each variable
        :param precision_list: list, precision of each variable
        :return:
                solver_ut._get_tensor(normalized_np): Tensor(n_numerical_inds,),
                    normalized numerical variables
        """
        vars_max = np.array([self.variables[i].upper for i in numerical_var_inds])
        vars_min = np.array([self.variables[i].lower for i in numerical_var_inds])
        if numerical_var_list.ndimension() == 1:
            bounded_np = np.array(
                [np.clip(k.item(), 0, 1) for kid, k in enumerate(numerical_var_list)]
            )
        else:
            bounded_np = np.array(
                [np.clip(k.numpy(), 0, 1) for k in numerical_var_list]
            )
        # adjust variable values to its pickable points
        raw_np = self.get_raw_vars(
            bounded_np,
            vars_max,
            vars_min,
            precision_list,
            normalized_ids=numerical_var_inds,
        )
        normalized_np = self.get_normalized_vars(
            raw_np, vars_max, vars_min, normalized_ids=numerical_var_inds
        )
        return solver_ut.get_tensor(normalized_np)

    # reuse code in UDAO
    def get_raw_vars(
        self,
        normalized_vars: np.ndarray,
        vars_max: np.ndarray,
        vars_min: np.ndarray,
        precision_list: list,
        normalized_ids: Optional[list] = None,
    ) -> np.ndarray:
        """
        denormalize the values of each variable
        :param normalized_vars: ndarray((bs, n_vars) or (n_vars,)), normalized variables
        :param vars_max: ndarray(n_vars,), maximum value of all variables
        :param vars_min: ndarray(n_vars,), minimum value of all variables
        :param precision_list: list, precision for all variables
        :param normalized_ids: list, indices of numerical variables (float and integer)
        :return: raw_vars: ndarray((bs, n_vars) or (n_vars,)), denormalized variables
        """
        vars_max = vars_max if normalized_ids is None else vars_max[normalized_ids]
        vars_min = vars_min if normalized_ids is None else vars_min[normalized_ids]
        precision_list = (
            precision_list
            if normalized_ids is None
            else np.array(precision_list)[normalized_ids].tolist()
        )
        vars = normalized_vars * (vars_max - vars_min) + vars_min
        print(vars)
        raw_vars = np.array(
            [c.round(p) for c, p in zip(vars.astype(float).T, precision_list)]
        ).T
        return raw_vars

    # reuse code in UDAO
    def get_normalized_vars(
        self,
        raw_vars: np.ndarray,
        vars_max: np.ndarray,
        vars_min: np.ndarray,
        normalized_ids: Optional[list] = None,
    ) -> np.ndarray:
        """
        normalize the values of each variable
        :param raw_vars: ndarray((bs, n_vars) or (n_vars, )),
            raw variable values (bounded with original lower and upper var_ranges)
        :param vars_max: ndarray(n_vars,), maximum value of all variables
        :param vars_min: ndarray(n_vars,), minimum value of all variables
        :param normalized_ids: list, indices fo numerical variables (float and integer)
        :return: normalized_vars, ndarray((bs, n_vars) or (n_vars, )),
            normalized variable values
        """
        vars_max = vars_max if normalized_ids is None else vars_max[normalized_ids]
        vars_min = vars_min if normalized_ids is None else vars_min[normalized_ids]
        normalized_vars = (raw_vars - vars_min) / (vars_max - vars_min)
        return normalized_vars

    def get_bounds(self, variables: List[Variable]) -> Tuple[np.ndarray, np.ndarray]:
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
        if not th.is_tensor(vars):  # type: ignore
            vars = solver_ut.get_tensor(vars)
        if vars.ndim == 1:
            obj_pred = cast(
                th.Tensor,
                self.objectives[obj_ind].function(
                    vars.reshape([1, vars.shape[0]]), wl_id
                ),
            )
        else:
            obj_pred = cast(th.Tensor, self.objectives[obj_ind].function(vars, wl_id))

        assert obj_pred.ndimension() == 2
        return obj_pred

    def _get_obj_pred_dict(
        self, wl_id: str, cst_dict: Dict, best_raw_vars: np.ndarray
    ) -> Dict:
        """
        get objective values
        :param wl_id: str, workload id, e.g. '1-7'
        :param cst_dict: dict, keys are objective names,
            values are bounds for each objective
        :param best_obj_dict: dict, keys are objective names,
            values are objective values,
            e.g. {'latency': 12406.1416015625, 'cores': 48.0}
        :param best_raw_vars: ndarray(n_vars,), variable values
        :return: obj_pred_dict: dict, keys are objective names,
            values are objective values
        """
        vars_max, vars_min = self.get_bounds(self.variables)
        vars_norm = self.get_normalized_vars(best_raw_vars, vars_max, vars_min)
        obj_pred_dict = {
            cst_obj: self._get_tensor_obj_pred(wl_id, vars_norm, obj_i)[0, 0].item()
            for obj_i, cst_obj in enumerate(cst_dict)
        }
        return obj_pred_dict

    ##################
    ## _check       ##
    ##################

    def _check_vars(self, vars_raw: np.ndarray) -> bool:
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
        if not (any(objective.name == obj_name for objective in self.objectives)):
            print(f"ERROR: objective {obj_name} is not found")
            return False
        return True

    # check violations of constraint functions
    def check_const_func_vio(self, wl_id: str, best_var: Optional[np.ndarray]) -> bool:
        """
        check whether the best variable values resulting
        in violation of constraint functions
        :param wl_id: str, workload id, e.g. '1-7'
        :param best_var: ndarray(n_vars, ), best variable values for each variable
        :return: bool, whether it returns feasible solutions (True) or not (False)
        """
        if best_var is None:
            return False

        tensor_best_var = solver_ut.get_tensor(best_var)

        if tensor_best_var.ndim == 1:
            tensor_best_var = tensor_best_var.reshape([1, best_var.shape[0]])
        const_loss = self._get_tensor_loss_const_funcs(wl_id, tensor_best_var)
        if const_loss <= 0:
            return True
        else:
            return False

    # check violations of objective value var_ranges
    # reuse code in UDAO
    def check_obj_bounds_vio(self, pred_dict: Optional[Dict], obj_bounds: Dict) -> bool:
        """
        check whether violating the objective value var_ranges
        :param pred_dict: dict, keys are objective names,
        values are objective values
        :param obj_bounds: dict, keys are objective names,
        values are lower and upper var_ranges of each objective value
        :return: True or False
        """
        if pred_dict is None:
            return False
        for obj, obj_pred in pred_dict.items():
            lower, upper = obj_bounds[obj]
            if lower.item() <= obj_pred <= upper.item():
                pass
            else:
                return False
        return True
