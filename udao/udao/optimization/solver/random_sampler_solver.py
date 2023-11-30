from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from ..concepts import EnumVariable, FloatVariable, IntegerVariable, Variable
from .sampler_solver import SamplerSolver


class RandomSampler(SamplerSolver):
    @dataclass
    class Params:
        n_samples_per_param: int
        "the number of samples per variable"

        seed: Optional[int] = None
        "random seed for generatino of samples"

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.n_samples_per_param = params.n_samples_per_param
        self.seed = params.seed

    def _process_variable(self, var: Variable) -> np.ndarray:
        """Generate samples of a variable"""
        if isinstance(var, FloatVariable):
            return np.random.uniform(var.lower, var.upper, self.n_samples_per_param)
        elif isinstance(var, IntegerVariable):
            return np.random.randint(
                var.lower, var.upper + 1, size=self.n_samples_per_param
            )
        elif isinstance(var, EnumVariable):
            inds = np.random.randint(0, len(var.values), size=self.n_samples_per_param)
            return np.array(var.values)[inds]
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(self, variables: Mapping[str, Variable]) -> Dict[str, np.ndarray]:
        """
        generate samples of variables

        Parameters:
        -----------
        variables: List[Variable],
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        Returns:
        --------
        Dict[str, np.ndarray]
            Dict with array of values for each variable
        """
        result_dict = {}

        np.random.seed(self.seed)
        for name, var in variables.items():
            result_dict[name] = self._process_variable(var)

        return result_dict