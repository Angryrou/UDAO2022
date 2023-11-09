from typing import Dict

import numpy as np

from ..utils.parameters import VarTypes
from .base_solver import BaseSolver


class RandomSampler(BaseSolver):
    def __init__(self, rs_params: Dict) -> None:
        """
        :param rs_params: int, the number of samples per variable
        """
        super().__init__()
        self.n_samples_per_param = rs_params["n_samples"]
        self.seed = rs_params["seed"]

    def _rand_float(self, lower: int, upper: int, n_samples: int) -> np.ndarray | None:
        """
        generate n_samples random float values within the lower and upper var_ranges
        :param lower: int, lower bound
        :param upper: int upper bound
        :param n_samples: int, the number of samples
        :return: ndarray(n_samples, ), n_samples random float
        """
        if lower > upper:
            return None
        else:
            scale = upper - lower
            out = np.random.rand(n_samples) * scale + lower
            return out

    def _get_input(self, var_ranges: np.ndarray, var_types: list) -> np.ndarray:
        """
        generate samples of variables
        :param var_ranges: array (n_vars,),
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        :param var_types: list,
            type of each variable
        :return: array,
            variables (n_samples * n_vars)
        """
        n_vars = var_ranges.shape[0]
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, values in enumerate(var_ranges):
            upper, lower = values[1], values[0]
            if (lower - upper) > 0:
                raise Exception(
                    f"ERROR: the lower bound of variable {i}"
                    " is greater than its upper bound!"
                )

            # randomly sample n_samples within the range
            if var_types[i] == VarTypes.FLOAT:
                x[:, i] = self._rand_float(lower, upper, self.n_samples_per_param)
            elif var_types[i] == VarTypes.INTEGER or var_types[i] == VarTypes.BOOL:
                x[:, i] = np.random.randint(
                    lower, upper + 1, size=self.n_samples_per_param
                )
            elif var_types[i] == VarTypes.ENUM:
                inds = np.random.randint(0, len(values), size=self.n_samples_per_param)
                x[:, i] = np.array(values)[inds]
            # TODO: extend to a matrix variable for the assignment problem in the future
            else:
                raise Exception(
                    f"Random-Sampler solver does not support variable type {var_types[i]}!"
                )
        return x
