from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .constraint import Constraint
from .objective import Objective
from .variable import Variable


@dataclass
class MOProblem:
    """Multi-objective optimization problem."""

    objectives: Sequence[Objective]
    """List of objectives to optimize"""
    variables: Dict[str, Variable]
    """Dictionary of variables to optimize"""
    constraints: Sequence[Constraint]
    """List of constraints to comply with"""
    input_parameters: Optional[Dict[str, Any]] = None
    """Dictionary of non-decision input parameters"""


@dataclass
class SOProblem:
    """Single-objective optimization problem."""

    objective: Objective
    """Objective to optimize"""
    variables: Dict[str, Variable]
    """Dictionary of variables to optimize"""
    constraints: Sequence[Constraint]
    """List of constraints to comply with"""
    input_parameters: Optional[Dict[str, Any]] = None
    """Dictionary of non-decision input parameters"""
