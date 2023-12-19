from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ...data.handler.data_processor import DataProcessor
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
    data_processor: Optional[DataProcessor] = None
    """Data processor to use for the problem"""
    input_parameters: Optional[Dict[str, Any]] = None
    """Dictionary of non-decision input parameters"""

    def __repr__(self) -> str:
        return (
            f"MOProblem(objectives={self.objectives}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )


@dataclass
class SOProblem:
    """Single-objective optimization problem."""

    objective: Objective
    """Objective to optimize"""
    variables: Dict[str, Variable]
    """Dictionary of variables to optimize"""
    constraints: Sequence[Constraint]
    """List of constraints to comply with"""
    data_processor: Optional[DataProcessor] = None
    """Data processor to use for the problem"""
    input_parameters: Optional[Dict[str, Any]] = None
    """Dictionary of non-decision input parameters"""

    def __repr__(self) -> str:
        return (
            f"SOProblem(objective={self.objective}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )
