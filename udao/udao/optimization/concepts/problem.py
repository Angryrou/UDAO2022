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

    def __repr__(self) -> str:
        return (
            f"MOProblem(objectives={self.objectives}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )


@dataclass
class SOProblem:
    objective: Objective
    variables: Dict[str, Variable]
    constraints: Sequence[Constraint]
    input_parameters: Optional[Dict[str, Any]] = None
    """Dictionary of non-decision input parameters"""

    def __repr__(self) -> str:
        return (
            f"SOProblem(objective={self.objective}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )
