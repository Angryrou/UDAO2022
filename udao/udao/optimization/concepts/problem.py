from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .constraint import Constraint
from .objective import Objective
from .variable import Variable


@dataclass
class MOProblem:
    """Multi-objective optimization problem."""

    objectives: Sequence[Objective]
    variables: Dict[str, Variable]
    constraints: Sequence[Constraint]
    input_parameters: Optional[Dict[str, Any]] = None
    """non-decision input parameters"""


@dataclass
class SOProblem:
    """Single-objective optimization problem."""

    objective: Objective
    variables: Dict[str, Variable]
    constraints: Sequence[Constraint]
    input_parameters: Optional[Dict[str, Any]] = None
    """non-decision input parameters"""
