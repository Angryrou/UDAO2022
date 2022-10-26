# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Multi-Objective Gradient Descent solver
#
# Created at 15/09/2022
# import torch

from optimization.solver.base_solver import BaseSolver
import utils.optimization.moo_utils as moo_ut
import utils.optimization.solver_utils as solver_ut
from utils.parameters import VarTypes

import time
import torch as th
import torch.optim as optim
from torch.multiprocessing import Pool
# from multiprocessing.pool import Pool
import numpy as np

SEED = 0
DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32

class MOGD(BaseSolver):
    def __init__(self, mogd_params: dict):
        '''
        initialize solver
        :param mogd_params: dict, parameters used in solver
        '''
        super().__init__()
        self.lr = mogd_params["learning_rate"]
        self.wd = mogd_params["weight_decay"]
        self.max_iter = mogd_params["max_iters"]
        self.patient = mogd_params["patient"]
        self.multistart = mogd_params["multistart"]
        self.process = mogd_params["processes"]
        self.stress = mogd_params["stress"]
        self.seed = mogd_params["seed"]

        self.device = th.device('cuda') if th.cuda.is_available() else th.device("cpu")
        self.dtype = th.float32

    def _problem(self, obj_funcs, opt_types, const_funcs, const_types):
        '''
        set up problem in solver
        :param obj_funcs: list, objective functions
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=" or "<", e.g. g1(x1, x2, ...) - c <= 0)
        :return:
        '''
        self.obj_funcs = obj_funcs
        self.opt_types = opt_types
        self.const_funcs = const_funcs
        self.const_types = const_types

    def single_objective_opt(self, obj, opt_obj_ind, var_types, var_range, precision_list, bs=1, verbose=False):
        '''
        solve (constrained) single-objective optimization
        :param obj: str, objective name
        :param opt_obj_ind: int, the index of objective to optimize
        :param var_types: list, variable types (float, integer, binary, enum)
        :param var_range: ndarray (n_vars,), the lower and upper var_ranges of non-ENUM variables, and values of ENUM variables
        :param bs: int, batch size
        :param verbose: bool, to print further information if needed
        :return:
                objs_list[idx]: float, objective value
                vars_list[idx].reshape([1, len(var_types)]): ndarray(1, n_vars), variable values
        '''

        th.manual_seed(self.seed)

        assert self.multistart >= 1

        conf_max, conf_min = self.get_bounds(var_range=var_range, var_types=var_types)

        meshed_categorical_vars = self.get_meshed_categorical_vars(var_types, var_range)
        numerical_var_inds = self.get_numerical_var_inds(var_types)
        categorical_var_inds = self.get_categorical_var_inds(var_types)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = [0]

        best_loss_list = []
        objs_list, vars_list = [], []
        for si in range(self.multistart):
            best_loss, best_obj, best_vars, iter_num = np.inf, np.inf, None, 0

            for bv in meshed_categorical_vars:
                bv_dict = {cind: bv[ind] for ind, cind in enumerate(categorical_var_inds)}
                numerical_var_list = th.rand(bs, len(numerical_var_inds), device=self.device, dtype=self.dtype,
                                                 requires_grad=True)
                optimizer = optim.Adam([numerical_var_list], lr=self.lr, weight_decay=self.wd)

                local_best_iter, local_best_loss, local_best_obj, local_best_var = 0, np.inf, None, None
                iter = 0

                for iter in range(self.max_iter):
                    vars_kernal = self._get_tensor_conf_cat(numerical_var_inds, numerical_var_list,
                                                     categorical_var_inds, bv_dict)
                    vars_kernal.data = solver_ut._get_tensor(self.get_raw_conf(vars_kernal.data.numpy().copy(), conf_max, conf_min, precision_list))
                    vars = vars_kernal
                    obj_pred = self._get_tensor_obj_pred(vars, opt_obj_ind)  # Nx1
                    loss, loss_id = self._loss_soo_minibatch(opt_obj_ind, obj_pred, vars)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_obj = obj_pred[loss_id].item()
                        if vars.ndim == 1:
                            local_best_var = vars.data.numpy()[loss_id].copy()
                        else:
                            local_best_var = vars.data.numpy()[loss_id, :].copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  # update parameters

                    constrained_numerical_var_list = self._get_tensor_numerical_constrained_vars(
                        numerical_var_list.data, numerical_var_inds, conf_max, conf_min, precision_list)
                    numerical_var_list.data = constrained_numerical_var_list

                    if iter > local_best_iter + self.patient:
                        # early stop
                        break

                    if verbose:
                        if iter % 10 == 0:
                            print(f'iteration {iter}, {obj}: {obj_pred[loss_id].item():.2f}')
                            print(vars)

                if verbose:
                    print(f'Finished at iteration {iter}, best local {obj} found as {local_best_obj:.5f}'
                          f' \nat iteration {local_best_iter},'
                          f' \nwith confs: {self.get_raw_conf(local_best_var, conf_max, conf_min, precision_list)}')

                iter_num += iter + 1

                if self.check_const_func_vio(local_best_var):
                    if local_best_loss < best_loss:
                        best_obj = local_best_obj
                        best_loss = local_best_loss
                        best_vars = local_best_var

            if self.check_const_func_vio(best_vars):
                if verbose:
                    print()
                    print("*" * 10)
                    print(f"get best {obj}: {best_obj} at {best_vars} with {iter_num} iterations, loss = {best_loss}")
            else:
                if verbose:
                    print("No valid solutions and variables found!")

            best_loss_list.append(best_loss)
            objs_list.append(best_obj)
            vars_list.append(best_vars)

        idx = np.argmin(best_loss_list)
        if vars_list[idx] is not None:
            vars = vars_list[idx].reshape([1, len(var_types)])
        else:
            vars = None
        return objs_list[idx], vars

    def constraint_so_opt(self, obj, opt_obj_ind, var_types, var_range, obj_bounds_dict, precision_list, verbose=False):
        '''
        solve single objective optimization constrained by objective values
        :param obj: str, name of objective to be optimized
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types
        :param var_range: ndarray (n_vars,), the lower and upper var_ranges of non-ENUM variables, and values of ENUM variables
        :param obj_bounds_dict: dict, keys are the name of objectives, values are the lower and upper var_ranges for each objective value
        :param precision_list: list, precision of each variable
        :param verbose: bool, to print further information if needed
        :return:
                objs_list[idx]: list, all objective values
                vars: ndarray(1, n_vars), variable values
        '''

        th.manual_seed(self.seed)
        assert self.multistart >= 1

        conf_max, conf_min = self.get_bounds(var_range, var_types)

        meshed_categorical_vars = self.get_meshed_categorical_vars(var_types, var_range)
        numerical_var_inds = self.get_numerical_var_inds(var_types)
        categorical_var_inds = self.get_categorical_var_inds(var_types)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = [0]

        best_loss_list = []
        objs_list, vars_list = [], []
        for si in range(self.multistart):
            best_loss, best_objs, best_vars, iter_num = np.inf, None, None, 0

            for bv in meshed_categorical_vars:
                bv_dict = {cind: bv[ind] for ind, cind in enumerate(categorical_var_inds)}
                numerical_var_list = th.rand(len(numerical_var_inds), device=self.device, dtype=self.dtype,
                                             requires_grad=True)
                optimizer = optim.Adam([numerical_var_list], lr=self.lr, weight_decay=self.wd)

                local_best_iter, local_best_loss, local_best_objs, local_best_var = 0, np.inf, None, None
                iter = 0

                for iter in range(self.max_iter):
                    vars_kernal = self._get_tensor_conf_cat(numerical_var_inds, numerical_var_list,
                                                            categorical_var_inds, bv_dict)

                    vars_kernal.data = solver_ut._get_tensor(
                        self.get_raw_conf(vars_kernal.data.numpy().copy(), conf_max, conf_min, precision_list))
                    vars = vars_kernal

                    objs_pred_dict = {cst_obj: self._get_tensor_obj_pred(vars, i) for i, cst_obj in enumerate(obj_bounds_dict)}
                    loss = self._loss_soo(vars, objs_pred_dict, obj_bounds_dict, target_obj_name=obj, target_obj_ind=opt_obj_ind)

                    if iter > 0 and loss.item() < local_best_loss:
                        local_best_loss = loss.item()
                        local_best_objs = {k: v.item() for k, v in objs_pred_dict.items()}
                        local_best_var = vars.data.numpy().copy()
                        local_best_iter = iter

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  # update parameters

                    constrained_numerical_var_list = self._get_tensor_numerical_constrained_vars(
                        numerical_var_list.data, numerical_var_inds, conf_max, conf_min, precision_list)
                    numerical_var_list.data = constrained_numerical_var_list

                    if iter > local_best_iter + self.patient:
                        # early stop
                        break

                    if verbose:
                        if iter % 10 == 0:
                            print(f'iteration {iter}, {obj}: {objs_pred_dict[obj].item():.2f}')
                            print(vars)

                if verbose:
                    print(f'Finished at iteration {iter}, best local {obj} found as {local_best_objs[obj]:.5f}'
                          f' \nat iteration {local_best_iter},'
                          f' \nwith confs: {self.get_raw_conf(local_best_var, conf_max, conf_min, precision_list)}')

                iter_num += iter + 1

                if self.check_const_func_vio(local_best_var) & self.check_obj_bounds_vio(local_best_objs, obj_bounds_dict):
                    if local_best_loss < best_loss:
                        best_objs = local_best_objs
                        best_loss = local_best_loss
                        best_vars = local_best_var

            if self.check_const_func_vio(best_vars) & self.check_obj_bounds_vio(best_objs, obj_bounds_dict):

                if verbose:
                    print()
                    print("*" * 10)
                    print(f"get best {obj}: {best_objs} at {best_vars} with {iter_num} iterations, loss = {best_loss}")
            else:
                if verbose:
                    print("No valid solutions and variables found!")

            best_loss_list.append(best_loss)
            objs_list.append(best_objs)
            vars_list.append(best_vars)

        idx = np.argmin(best_loss_list)

        if vars_list[idx] is not None:
            objs = list(objs_list[idx].values())
            vars = vars_list[idx].reshape([1, len(var_types)])
        else:
            objs, vars = None, None

        return objs, vars

    def constraint_so_parallel(self, obj, opt_obj_ind, var_types, var_range, precision_list, cell_list, verbose=False):
        '''
        solve the single objective optimization constrained by objective values parallelly
        :param obj: str, name of objective to be optimized
        :param opt_obj_ind: int, index of objective to be optimized
        :param var_types: list, variable types
        :param var_range: ndarray (n_vars,), the lower and upper var_ranges of non-ENUM variables, and values of ENUM variables
        :param precision_list: list, precision of each variable
        :param cell_list: list, each element is a dict to indicate the var_ranges of objective values
        :param verbose: bool, to print further information if needed
        :return:
                po_objs_list: list, each element is a Pareto solution
                po_vars_list: list, each element is the variable values corresponding to the Pareto solution
        '''

        th.manual_seed(self.seed)

        # generate conf_list for constraint_so_opt
        arg_list = [(obj, opt_obj_ind, var_types, var_range, obj_bounds_dict, precision_list)
                    for obj_bounds_dict in cell_list]

        # call self.constraint_so_opt parallely
        th.multiprocessing.set_start_method('fork')
        with Pool(processes=self.process) as pool:
            ret_list = pool.starmap(self.constraint_so_opt, arg_list)

        # sort out the output
        po_objs_list = [solution[0] for solution in ret_list if solution[0] is not None]
        po_vars_list = [solution[1].tolist()[0] for solution in ret_list if solution[1] is not None]

        return po_objs_list, po_vars_list

    def get_meshed_categorical_vars(self, var_types, var_range):
        '''
        get combinations of all categorical (binary, enum) variables
        # reuse code in UDAO
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: meshed_cv_value: ndarray, categorical(binary, enum) variables
        '''

        categorical_var_inds = self.get_categorical_var_inds(var_types)
        if len(categorical_var_inds) == 0:
            return None
        else:
            cv_value_list = [value for value in var_range[categorical_var_inds]]
            meshed_cv_value_list = [x_.reshape(-1, 1) for x_ in np.meshgrid(*cv_value_list)]
            meshed_cv_value = np.concatenate(meshed_cv_value_list, axis=1)
            return meshed_cv_value

    def get_categorical_var_inds(self, var_types):
        '''
        get indices of categorical (binary, enum) variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: categorical_var_inds: list, indices of categorical (binary, enum) variables
        '''
        # categorical_var_inds = [ind for ind, v_ in enumerate(var_types) if (v_ == 'BINARY') or (v_ == 'ENUM')]
        categorical_var_inds = [ind for ind, v_ in enumerate(var_types) if (v_ == VarTypes.BOOL) or (v_ == VarTypes.ENUM)]
        return categorical_var_inds

    def get_numerical_var_inds(self, var_types):
        '''
        get indices of numerical (float, integer) variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: numerical_var_inds: list, indices of numerical (float, integer) variables
        '''
        # numerical_var_inds = [ind for ind, v_ in enumerate(var_types) if (v_ == 'FLOAT') or (v_ == 'INTEGER')]
        numerical_var_inds = [ind for ind, v_ in enumerate(var_types) if (v_ == VarTypes.FLOAT) or (v_ == VarTypes.INTEGER)]
        return numerical_var_inds

    def _get_tensor_conf_cat(self, numerical_var_inds, numerical_var_list, categorical_var_inds,
                             cv_dict):
        '''
        concatenate values of numerical(float, integer) and categorical(binary, enum) variables together
        (reuse code in UDAO)
        :param numerical_var_inds: list, indices of numerical variables
        :param numerical_var_list: tensor((bs, len(numerical_var_inds)) or (numerical_var_inds,)), values of numercial variables
        :param categorical_var_inds: list, indices of (binary, enum) variables
        :param cv_dict: dict(key: indices of (binary, enum) variables, value: value of (binary, enum) variables), indices and values of bianry variables
        :return: conf: tensor((bs, len(numerical_var_inds)) or (numerical_var_inds,)), values of all variables
        '''
        #
        # conf is the variables
        target_len = len(numerical_var_inds) + len(categorical_var_inds)
        to_concat = []
        ck_ind = 0
        if numerical_var_list.ndimension() == 1:
            for i in range(target_len):
                if i in set(numerical_var_inds):
                    to_concat.append(numerical_var_list[i - ck_ind: i + 1 - ck_ind])
                elif i in set(categorical_var_inds):
                    to_concat.append(solver_ut._get_tensor([cv_dict[i]]))
                    ck_ind += 1
                else:
                    raise Exception(f'unsupported type in var {i}')
            conf = th.cat(to_concat)
        else:
            n_batch = numerical_var_list.shape[0]
            for i in range(target_len):
                if i in set(numerical_var_inds):
                    to_concat.append(numerical_var_list[:, i - ck_ind: i + 1 - ck_ind])
                elif i in set(categorical_var_inds):
                    to_concat.append(
                        th.ones((n_batch, 1), device=self.device, dtype=self.dtype) * cv_dict[i])
                    ck_ind += 1
                else:
                    raise Exception(f'unsupported type in var {i}')
            conf = th.cat(to_concat, dim=1)
            assert conf.shape[0] == numerical_var_list.shape[0] and conf.shape[1] == target_len
        return conf

    def _get_tensor_obj_pred(self, vars, obj_ind):
        '''
        get objective values
        :param vars: tensor ((bs, n_vars) or (n_vars, )), variables, where bs is batch_size
        :param obj_ind: int, the index of objective to optimize
        :return: obj_pred: tensor(1,1), the objective value
        '''
        if not th.is_tensor(vars):
            vars = solver_ut._get_tensor(vars)
        if vars.ndim == 1:
            obj_pred = self.obj_funcs[obj_ind](vars.reshape([1, vars.shape[0]]))
        else:
            obj_pred = self.obj_funcs[obj_ind](vars)


        return obj_pred

    ##################
    ## _loss        ##
    ##################
    def _get_tensor_loss_const_funcs(self, vars):
        '''
        compute loss of the values of each constraint function
        :param vars: tensor ((bs, n_vars) or (n_vars, )), variables, where bs is batch_size
        :return: const_loss: tensor (Tensor:()), loss of the values of each constraint function
        '''
        # vars: a tensor
        # get loss for constraint functions defined in the problem setting
        if vars.ndim == 1:
            vars = vars.reshape([1, vars.shape[0]])
        const_violation = th.tensor(0, device=self.device, dtype=self.dtype)
        for i, const_func in enumerate(self.const_funcs):
            if self.const_types[i] == "<=":
                const_violation = th.relu(const_violation + const_func(vars))
            elif self.const_types[i] == "==":
                const_violation1 = th.relu(const_violation + const_func(vars))
                const_violation2 = th.relu(const_violation + (const_func(vars)) * (-1))
                const_violation = const_violation1 + const_violation2
            elif self.const_types[i] == ">=":
                const_violation = th.relu(const_violation + (const_func(vars)) * (-1))
            else:
                raise Exception(f"{self.const_types[i]} is not supported!")

        if const_violation.sum() != 0:
            const_loss = const_violation ** 2 + 1e5
        else:
            const_loss = th.tensor(0, device=self.device, dtype=self.dtype)

        # print(f"const_loss: {const_loss}")
        return const_loss

    def _loss_soo_minibatch(self, obj_ind, obj_pred, vars):
        '''
        compute loss
        :param obj_ind: int, the index of the objective to be optimized
        :param obj_pred: tensor, objective value(prediction)
        :param vars: tensor ((bs, n_vars) or (n_vars, )), variables, where bs is batch_size
        :return: [tensor, tensor], minimum loss and its index
        '''
        loss = (obj_pred ** 2) * moo_ut._get_direction(self.opt_types, obj_ind)
        loss = loss + self._get_tensor_loss_const_funcs(vars)
        return th.min(loss), th.argmin(loss)

    def _loss_soo(self, vars, pred_dict, obj_bounds, target_obj_name, target_obj_ind):
        '''
        compute loss constrained by objective values
        # reuse code in UDAO
        :param vars: tensor ((bs, n_vars) or (n_vars, )), variables, where bs is batch_size
        :param pred_dict: dict, keys are objective names, values are objective values
        :param obj_bounds: dict, keys are objective names, values are lower and upper var_ranges of each objective value
        :param target_obj_name: str, the name of the objective to be optimized
        :param target_obj_ind: int, the index of target_obj_name
        :return:
                loss: tensor (Tensor())
        '''
        loss = th.tensor(0, device=self.device, dtype=self.dtype)

        for cst_obj, [lower, upper] in obj_bounds.items():
            assert pred_dict[cst_obj].shape[0] == 1
            obj_pred_raw = pred_dict[cst_obj].sum() # (1,1)

            if upper != lower:
                norm_cst_obj_pred = (obj_pred_raw - lower) / (upper - lower)  # scaled
                add_loss = th.tensor(0, device=self.device, dtype=self.dtype)
                if cst_obj == target_obj_name:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
                    else:
                        add_loss += norm_cst_obj_pred * moo_ut._get_direction(self.opt_types, target_obj_ind)
                else:
                    if norm_cst_obj_pred < 0 or norm_cst_obj_pred > 1:
                        add_loss += (norm_cst_obj_pred - 0.5) ** 2 + self.stress
            else:
                add_loss = (obj_pred_raw - upper) ** 2 + self.stress
            loss = loss + add_loss
        loss = loss + self._get_tensor_loss_const_funcs(vars)
        return loss

    def _get_tensor_numerical_constrained_vars(self, numerical_var_list, numerical_var_inds,
                                               conf_max, conf_min, precision_list):
        '''
        make the values of numerical variables within their range
        :param numerical_var_list: tensor ((bs, len(numerical_var_inds) or (len(numerical_var_ids), ), values of numerical variables (FLOAT and INTEGER)
        :param numerical_var_inds: list, indices of numerical variables (float and integer)
        :param conf_max: ndarray(n_vars, ), upper var_ranges of each variable
        :param conf_min: ndarray(n_vars, ), lower var_ranges of each variable
        :param precision_list: list, precision of each variable
        :return:
        '''
        conf_cons_max_norm = (conf_max - conf_min) / (conf_max - conf_min)
        conf_cons_min_norm = (conf_min - conf_min) / (conf_max - conf_min)
        conf_cons_max_ = conf_cons_max_norm[numerical_var_inds]
        conf_cons_min_ = conf_cons_min_norm[numerical_var_inds]
        if numerical_var_list.ndimension() == 1:
            bounded_np = np.array([self.get_bounded(k.item(), lower=conf_cons_min_[kid], upper=conf_cons_max_[kid])
                                   for kid, k in enumerate(numerical_var_list)])
        else:
            bounded_np = np.array([self.get_bounded(k.numpy(), lower=conf_cons_min_, upper=conf_cons_max_)
                                   for k in numerical_var_list])
        # adjust variable values to its pickable points
        raw_np = self.get_raw_conf(bounded_np, conf_max, conf_min, precision_list, normalized_ids=numerical_var_inds)
        normalized_np = self.get_normalized_conf(raw_np, conf_max, conf_min, normalized_ids=numerical_var_inds)
        return solver_ut._get_tensor(normalized_np)

    # reuse code in UDAO
    def get_bounded(self, k, lower=0.0, upper=1.0):
        '''
        make normalized variable values bounded within the range 0 and 1
        :param k: ndarray(len(numerical_var_inds), ), normalized value of numerical variables (float and integer)
        :param lower: ndarray(len(numerical_var_inds), ), 0
        :param upper: ndarray(len(numerical_var_inds), ), 1
        :return: k: ndarray(len(numerical_var_inds), ), normalized value bounded within range (0, 1) for each variable
        '''
        k = np.maximum(k, lower)
        k = np.minimum(k, upper)
        return k

    # reuse code in UDAO
    def get_raw_conf(self, normalized_conf, conf_max, conf_min, precision_list, normalized_ids=None):
        '''
        denormalize the values of each variable
        :param normalized_conf: ndarray((bs, n_vars) or (n_vars,)), normalized variables
        :param conf_max: ndarray(n_vars,), maximum value of all variables
        :param conf_min: ndarray(n_vars,), minimum value of all variables
        :param precision_list: list, precision for all variables
        :param normalized_ids: list, indices of numerical variables (float and integer)
        :return: raw_conf: ndarray((bs, n_vars) or (n_vars,)), denormalized variables
        '''
        conf_max = conf_max if normalized_ids is None else conf_max[normalized_ids]
        conf_min = conf_min if normalized_ids is None else conf_min[normalized_ids]
        precision_list = precision_list if normalized_ids is None else np.array(precision_list)[normalized_ids].tolist()
        # precision_list = precision_list if normalized_ids is None else precision_list[normalized_ids]
        conf = normalized_conf * (conf_max - conf_min) + conf_min
        # raw_conf = np.array([c.round(p) for c, p in zip(conf.T, precision_list)]).T
        raw_conf = np.array([c.round(p) for c, p in zip(conf.astype(float).T, precision_list)]).T
        return raw_conf

    # reuse code in UDAO
    def get_normalized_conf(self, raw_conf, conf_max, conf_min, normalized_ids=None):
        '''
        normalize the values of each variable
        :param raw_conf: ndarray((bs, n_vars) or (n_vars, )), raw variable values (bounded with original lower and upper var_ranges)
        :param conf_max: ndarray(n_vars,), maximum value of all variables
        :param conf_min: ndarray(n_vars,), minimum value of all variables
        :param normalized_ids: list, indices fo numerical variables (float and integer)
        :return: normalized_conf, ndarray((bs, n_vars) or (n_vars, )), normalized variable values
        '''
        conf_max = conf_max if normalized_ids is None else conf_max[normalized_ids]
        conf_min = conf_min if normalized_ids is None else conf_min[normalized_ids]
        normalized_conf = (raw_conf - conf_min) / (conf_max - conf_min)
        return normalized_conf

    # check violations of constraint functions
    def check_const_func_vio(self, best_var):
        '''
        check whether the best variable values resulting in violation of constraint functions
        :param best_var: ndarray(n_vars, ), best variable values for each variable
        :return:
        '''
        if best_var is None:
            return False

        if not th.is_tensor(best_var):
            best_var = solver_ut._get_tensor(best_var)

        if best_var.ndim == 1:
            best_var = best_var.reshape([1, best_var.shape[0]])
        const_loss = self._get_tensor_loss_const_funcs(best_var)
        if const_loss <= 0:
            return True
        else:
            return False

    # check violations of objective value var_ranges
    # reuse code in UDAO
    def check_obj_bounds_vio(self, pred_dict, obj_bounds):
        '''
        check whether violating the objective value var_ranges
        :param pred_dict: dict, keys are objective names, values are objective values
        :param obj_bounds: dict, keys are objective names, values are lower and upper var_ranges of each objective value
        :return: True or False
        '''
        if pred_dict is None:
            return False
        for obj, obj_pred in pred_dict.items():
            lower, upper = obj_bounds[obj]
            if lower.item() <= obj_pred <= upper.item():
                pass
            else:
                return False
        return True

    def get_bounds(self, var_range, var_types):
        '''
        get min and max values for each variable
        :param var_range: ndarray (n_vars,), the lower and upper var_ranges of non-ENUM variables, and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return: minimum and maximum values of all variables
                ndarray(n_vars,): the maximum values of all variables
                ndarray(n_vars,): the minimum values of all variables
        '''

        var_max, var_min = [], []
        for var_bound, var_type in zip(var_range, var_types):
            if (var_type == VarTypes.FLOAT) or (var_type == VarTypes.INTEGER) or (var_type == VarTypes.BOOL):
                var_max.append(var_bound[1])
                var_min.append(var_bound[0])
            elif var_type == VarTypes.ENUM:
                var_max.append(max(var_bound))
                var_min.append(min(var_bound))
            else:
                Exception(f"Variable type {var_type} is not supported!")

        return np.array(var_max), np.array(var_min)

