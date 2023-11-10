from typing import List, Optional

import numpy as np
from attr import dataclass
from udao.optimization.concepts.variable import (
    EnumVariable,
    FloatVariable,
    IntegerVariable,
    Variable,
)

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

    def _get_input(self, variables: List[Variable]) -> np.ndarray:
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
        n_vars = len(variables)
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, var in enumerate(variables):
            # randomly sample n_samples within the range
            if isinstance(var, FloatVariable):
                x[:, i] = np.random.uniform(
                    var.lower, var.upper, self.n_samples_per_param
                )
            elif isinstance(var, IntegerVariable):
                x[:, i] = np.random.randint(
                    var.lower, var.upper + 1, size=self.n_samples_per_param
                )
            elif isinstance(var, EnumVariable):
                inds = np.random.randint(
                    0, len(var.values), size=self.n_samples_per_param
                )
                x[:, i] = np.array(var.values)[inds]
            # TODO: extend to a matrix variable for the assignment problem in the future
            else:
                raise Exception(
                    "Random-Sampler solver does not"
                    f" support variable type {type(var)}!"
                )
        return x
