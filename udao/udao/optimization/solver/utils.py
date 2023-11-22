from typing import List, Optional

import numpy as np

from ..concepts import Constraint


def filter_on_constraints(
    wl_id: Optional[str], input_vars: np.ndarray, constraints: List[Constraint]
) -> np.ndarray:
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

    available_indices = np.array(range(len(input_vars)))
    for constraint in constraints:
        const_values = constraint.function(input_vars, wl_id=wl_id)
        if constraint.type == "<=":
            compliant_indices = np.where(const_values <= 0)
        elif constraint.type == ">=":
            compliant_indices = np.where(const_values >= 0)
        else:
            compliant_indices = np.where(const_values == 0)
        available_indices = np.intersect1d(compliant_indices, available_indices)
    filtered_vars = input_vars[available_indices]
    return filtered_vars
