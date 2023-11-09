from typing import Optional

import numpy as np
from attr import dataclass

from ..utils.parameters import VarTypes
from .base_solver import BaseSolver


class RandomSampler(BaseSolver):
    @dataclass
    class Params:
        n_samples_per_param: int
        seed: Optional[int] = None

    def __init__(self, params: Params) -> None:
        """
        :param rs_params: int, the number of samples per variable
        """
        super().__init__()
        self.n_samples_per_param = params.n_samples_per_param
        self.seed = params.seed

    def _get_input(self, var_ranges: list, var_types: list) -> np.ndarray:
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
        n_vars = len(var_ranges)
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
                x[:, i] = np.random.uniform(lower, upper, self.n_samples_per_param)
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
                    "Random-Sampler solver does not"
                    " support variable type {var_types[i]}!"
                )
        return x
