from typing import Any, Dict, List, Optional

import numpy as np

from ..concepts import Constraint


def filter_on_constraints(
    input_parameters: Optional[Dict[str, Any]],
    input_vars: Dict[str, np.ndarray],
    constraints: List[Constraint],
) -> Dict[str, np.ndarray]:
    """Keep only input variables that don't violate constraints

    Parameters:
    -----------
    wl_id : str | None
        workload id
    input_vars : np.ndarray
        input variables
    constraints : List[Constraint]
        constraint functions
    """
    if not constraints:
        return input_vars

    available_indices = np.arange(len(next(iter(input_vars.values()))))
    for constraint in constraints:
        const_values = constraint.function(
            input_vars, input_parameters=input_parameters
        )

        if constraint.type == "<=":
            compliant_indices = np.where(const_values <= 0)
        elif constraint.type == ">=":
            compliant_indices = np.where(const_values >= 0)
        else:
            compliant_indices = np.where(const_values == 0)
        available_indices = np.intersect1d(compliant_indices, available_indices)
    filtered_vars = {k: v[available_indices] for k, v in input_vars.items()}
    return filtered_vars
