from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from . import Constraint, Objective, Variable


@dataclass
class MOProblem:
    objectives: Sequence[Objective]
    variables: Dict[str, Variable]
    constraints: Sequence[Constraint]
    input_parameters: Optional[Dict[str, Any]] = None


@dataclass
class SOProblem:
    objective: Objective
    variables: Dict[str, Variable]
    constraints: Sequence[Constraint]
    input_parameters: Optional[Dict[str, Any]] = None
