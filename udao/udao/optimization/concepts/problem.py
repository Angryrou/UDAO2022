from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .constraint import Constraint
from .objective import Objective
from .variable import Variable


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
