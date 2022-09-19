# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 15/09/2022
import torch

from optimization.solver.base_solver import BaseSolver

from torch.multiprocessing import Pool
import time
import torch as th
import torch.optim as optim

from abc import ABCMeta, abstractmethod
import numpy as np

SEED = 0
DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DTYPE = th.float32

class MOGD(BaseSolver):
    def __init__(self, mogd_params, bounds, debug):
        super().__init__()
        self.lr, self.wd, self.max_iter = mogd_params["lr"], mogd_params["wd"], mogd_params["max_iter"]
        self.patient, self.multistart, self.process = \
            mogd_params["patient"], mogd_params["multistart"], mogd_params["processes"]

        self.debug = True
        self.bounds = bounds
        self.device = th.device('cuda') if th.cuda.is_available() else th.device("cpu")
        self.obj_ref = 1000

    def predict(self, vars, obj, n_objs):
        ## should be tensor
        objs_pred = []
        for i in range(n_objs):
            obj = self._obj_function(vars, obj=f"obj_{i + 1}")
            objs_pred.append(obj)

        return th.stack(objs_pred).T

    def single_objective_opt(self, obj, opt_obj_ind, var_types, n_vars, n_objs, bs=16, verbose=False):

        th.manual_seed(SEED)

        ## initialize
        best_loss, best_obj, best_vars = np.inf, None, None
        num_iters = 0

        for si in range(self.multistart):
            vars_kernel = th.rand(bs, n_vars, requires_grad=True, device=self.device)
            optimizer = optim.Adam([vars_kernel], lr=self.lr, weight_decay=self.wd)

            local_best_iter, local_best_loss, local_best_obj, local_best_theta = 0, np.inf, None, None
            iter = 0

            for iter in range(self.max_iter):

                # should be tensor
                vars = self._vars_inv_norm(vars_kernel)
                vars = self._vars_round_to_range(var_types, vars)
                objs_pred = self.predict(vars, obj, n_objs)

                # add loss for violating the constraints
                loss, loss_id = self._loss_so(objs_pred, opt_obj_ind, vars)

                if iter > 0 and loss < local_best_loss:
                    with th.no_grad():
                        local_best_loss = loss.detach().cpu().item()
                        local_best_obj = [objs_pred[:, i][loss_id].detach().cpu().item() for i in range(n_objs)]
                        local_best_vars = vars.detach().clone()
                        local_best_iter = iter

                if self.debug:
                    if (iter + 1) % 10 == 0:
                        print(f'iteration {si}-{iter + 1}, '
                              f'loss={loss.item():.3e}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter > local_best_iter + self.patient:
                    # early stop
                    break

            with th.no_grad():
                const_violation = self._const_function(vars)
                feasible_vars_flags = [all(x) for x in const_violation.le(0)]  # True means the variable vector is feasible
                if not any(feasible_vars_flags): # all flags are False
                    if self.debug:
                        print(f'Finished at iteration {si}-{iter}, not found')
                    continue

            if self.debug:
                print(f'Finished at iteration {si}-{iter}, '
                      f'at iteration {si}-{local_best_iter},')

            num_iters += (iter + 1)

            if local_best_loss < best_loss:
                best_obj = local_best_obj
                best_loss = local_best_loss
                best_vars = local_best_vars.cpu().numpy()

        if best_vars is None:
            return [None] * n_objs, None

        return best_obj, best_vars

    def constraint_so_opt(self, obj, opt_obj_ind, var_types, n_vars, n_objs, lower, upper, bs=16, verbose=False):
        th.manual_seed(SEED)

        ## initialize
        best_loss, best_obj, best_vars = np.inf, None, None
        num_iters = 0

        for si in range(self.multistart):
            vars_kernel = th.rand(bs, n_vars, requires_grad=True, device=self.device)
            optimizer = optim.Adam([vars_kernel], lr=self.lr, weight_decay=self.wd)

            local_best_iter, local_best_loss, local_best_obj, local_best_theta = 0, np.inf, None, None
            iter = 0

            for iter in range(self.max_iter):

                # should be tensor
                vars = self._vars_inv_norm(vars_kernel)
                vars = self._vars_round_to_range(var_types, vars)
                objs_pred = self.predict(vars, obj, n_objs)

                # add loss for violating the constraints
                loss, loss_id = self._loss_co(objs_pred, opt_obj_ind, vars, lower, upper)

                if iter > 0 and loss < local_best_loss:
                    with th.no_grad():
                        local_best_loss = loss.detach().cpu().item()
                        local_best_obj = [objs_pred[:, i][loss_id].detach().cpu().item() for i in range(n_objs)]
                        local_best_vars = vars.detach().clone()
                        local_best_iter = iter

                if self.debug:
                    if (iter + 1) % 10 == 0:
                        print(f'iteration {si}-{iter + 1}, '
                              f'loss={loss.item():.3e}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter > local_best_iter + self.patient:
                    # early stop
                    break

            with th.no_grad():
                const_violation = self._const_function(vars)
                feasible_vars_flags = [all(x) for x in
                                       const_violation.le(0)]  # True means the variable vector is feasible
                if not any(feasible_vars_flags):  # all flags are False
                    if self.debug:
                        print(f'Finished at iteration {si}-{iter}, not found')
                    continue

            if self.debug:
                print(f'Finished at iteration {si}-{iter}, '
                      f'at iteration {si}-{local_best_iter},')

            num_iters += (iter + 1)

            if local_best_loss < best_loss:
                best_obj = local_best_obj
                best_loss = local_best_loss
                best_vars = local_best_vars.cpu().numpy()

        if best_vars is None:
            return [None] * n_objs, None

        return best_obj, best_vars

    def _vars_inv_norm(self, vars_kernel):

        vars_min, vars_max = self.bounds[:, 0], self.bounds[:, 1]

        if vars_min.min() == -np.inf:
            neg_inf_inds = np.where(vars_min == -np.inf)
            #todo:
            vars_min[neg_inf_inds] = -1e1

        if vars_max.max() == np.inf:
            pos_inf_inds = np.where(vars_max== np.inf)
            # todo:
            vars_max[pos_inf_inds] = 1e1

        lower = self._get_tensor(vars_min)
        upper = self._get_tensor(vars_max)

        return lower + (upper - lower) * vars_kernel

    def _vars_round_to_range(self, var_types, vars):
        # find the vars with "int" or "categorical" types
        not_float_inds = np.where(np.array(var_types) != "float")
        if not_float_inds[0].size != 0:
            vars[not_float_inds] = vars[not_float_inds].round()

        return vars

    def _loss_so(self, obj_pred, opt_obj_ind, vars):
        const_violation = self._const_function(vars)
        mask = const_violation.ge(0)
        const_loss = th.masked_select(const_violation, mask).sum() * 100

        #todo: to choose a ref value rather than hard-coded
        loss = obj_pred[:, opt_obj_ind] / self.obj_ref + const_loss

        return th.min(loss), th.argmin(loss)

    def _loss_co(self, obj_pred, opt_obj_ind, vars, lower, upper):

        # violation of constraints from problem settings
        const_violation = self._const_function(vars)
        mask = const_violation.ge(0)
        const_loss = th.masked_select(const_violation, mask).sum() * 100

        # violation of bounds for middle point probe
        n_objs = obj_pred.size()[1]
        bs = obj_pred.size()[0]
        vio_mid_sum = 0
        for i in range(n_objs):
            if i == opt_obj_ind:
                continue
            else:
                vio_mid_upper = th.relu(obj_pred[:, i].reshape(bs, 1) - self._get_tensor(upper[i]))
                vio_mid_lower = th.relu(self._get_tensor(lower[i]) - obj_pred[:, i].reshape(bs, 1))
                vio_mid_sum += (vio_mid_upper.sum() + vio_mid_lower.sum()) * 100

        # todo: to choose a ref value rather than hard-coded as 1000
        loss = obj_pred[:, opt_obj_ind] / self.obj_ref + const_loss + vio_mid_sum

        return th.min(loss), th.argmin(loss)

    def _obj_function(self, vars, obj):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
        # Chankong and Haimes function:
        ## minimize:
        ##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
        ##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
        ## subject to:
        ##          g1(x, y) = x * x + y * y <= 225
        ##          g2(x, y) = x - 3 * y + 10 <= 0
        ##          x in [-20, inf], y in [-inf, 20]

        ## should be tensor
        if obj == "obj_1":
            value = 2 + (vars[:, 0] - 2) * (vars[:, 0] - 2) + (vars[:, 1] - 1) * (vars[:, 1] - 1)
        elif obj == "obj_2":
            value = 9 * vars[:, 0] - (vars[:, 1] - 1) * (vars[:, 1] - 1)
        else:
            raise ValueError(obj)
        return value

    def _const_function(self, vars):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
        # Chankong and Haimes function:
        ## minimize:
        ##          f1(x, y) = 2 + (x - 2) * (x - 2) + (y - 1) * (y - 1)
        ##          f2(x, y) = 9 * x - (y - 1) * (y - 1)
        ## subject to:
        ##          g1(x, y) = x * x + y * y <= 225
        ##          g2(x, y) = x - 3 * y + 10 <= 0
        ##          x in [-20, inf], y in [-inf, 20]

        ## add constraints
        # each g1 value shows the constraint violation
        g1 = vars[:, 0] * vars[:, 0] + vars[:, 1] * vars[:, 1] - 225
        g2 = vars[:, 0] - 3 * vars[:, 1] + 10 - 0

        # ## return array type
        # return np.hstack([g1.reshape([g1.shape[0], 1]), g2.reshape([g2.shape[0], 1])])

        # should be tensor
        return th.stack([g1, g2]).T

    def _get_tensor(self, x, dtype=None, device=None, requires_grad=False):
        dtype = th.float32
        device = DEFAULT_DEVICE if device is None else device

        return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)




