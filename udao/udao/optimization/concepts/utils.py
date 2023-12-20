from typing import Any, Dict, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import torch as th

from ...data.extractors.tabular_extractor import TabularFeatureExtractor
from ...data.handler.data_processor import DataProcessor
from ...data.iterators.base_iterator import BaseIterator, UdaoIterator
from ...utils.interfaces import UdaoInput

InputVariables = Union[Dict[str, np.ndarray], Dict[str, Any]]
InputParameters = Optional[Dict[str, Any]]


class UdaoFunction(Protocol):
    """A function that can be used for a constraint or an objective. called with:
    - input_variables: the variables inputs
    - input_parameters: the non-decision inputs
    """

    def __call__(
        self,
        input_variables: InputVariables,
        input_parameters: InputParameters = None,
    ) -> th.Tensor:
        ...


class ModelComponent:
    """A wrapper class for model and data_processor, to make the model callable"""

    def __init__(self, data_processor: DataProcessor, model: th.nn.Module) -> None:
        self.data_processor = data_processor
        self.model = model

    def process_data(
        self,
        input_variables: InputVariables,
        input_parameters: InputParameters = None,
    ) -> Tuple[Any, BaseIterator]:
        """Derive the batch input from the input dict."""
        return derive_processed_input(
            self.data_processor,
            input_parameters=input_parameters,
            input_variables=input_variables,
        )

    def __call__(
        self,
        input_variables: InputVariables,
        input_parameters: InputParameters = None,
    ) -> th.Tensor:
        input_data, _ = self.process_data(
            input_parameters=input_parameters, input_variables=input_variables
        )
        return self.model(input_data)

    def to(self, device: th.device) -> None:
        self.model.to(device)


class InaccurateModel(th.nn.Module):
    def __init__(
        self, model: th.nn.Module, std_func: th.nn.Module, alpha: float
    ) -> None:
        self.model = model
        self.std_func = std_func
        self.alpha = alpha

    def forward(self, x: th.Tensor) -> th.Tensor:
        std = self.std_func(x)
        return self.model(x) + self.alpha * std


def derive_unprocessed_input(
    input_variables: InputVariables,
    input_parameters: InputParameters = None,
    device: Optional[th.device] = None,
) -> Tuple[Dict[str, th.Tensor], Dict[str, Any]]:
    """Derive the input data from the input values

    Parameters
    ----------
    input_variables : InputVariables
        _description_
    input_parameters : InputParameters, optional
        _description_, by default None
    device : th.device, optional
        _description_, by default th.device("cpu")
    dtype : th.dtype, optional
        _description_, by default th.float32

    Returns
    -------
    Tuple[Dict[str, th.Tensor], Dict[str, Any]]
        _description_
    """
    variable_sample = input_variables[list(input_variables.keys())[0]]
    if isinstance(variable_sample, np.ndarray):
        n_items = len(variable_sample)
    else:
        n_items = 1
        input_variables = {k: np.array([v]) for k, v in input_variables.items()}
    input_parameters_values = {
        k: th.tensor([v] * n_items) for k, v in (input_parameters or {}).items()
    }
    return {
        name: th.tensor(value, device=device) for name, value in input_variables.items()
    }, input_parameters_values


def derive_processed_input(
    data_processor: DataProcessor[UdaoIterator],
    input_variables: InputVariables,
    input_parameters: InputParameters = None,
    device: Optional[th.device] = None,
) -> Tuple[UdaoInput, BaseIterator]:
    """Derive the batch input from the input dict

    Parameters
    ----------
    input_non_decision : Dict[str, Any]
        The fixed values for the non-decision inputs

    input_variables : Dict[str, np.ndarray] | Dict[str, Any]
        The values for the variables inputs:
        - a batch of all values if the variable is a numpy array
        - a batch of a single value if the variable is a single value

    Returns
    -------
    Any
        The batch input for the model
    """
    variable_sample = input_variables[list(input_variables.keys())[0]]
    if isinstance(variable_sample, np.ndarray):
        n_items = len(input_variables[list(input_variables.keys())[0]])
    else:
        n_items = 1
        input_variables = {k: [v] for k, v in input_variables.items()}
    keys = [f"{i}" for i in range(n_items)]
    objective_values = {}
    # hack to get dummy objective values when using a tabular extractor
    if (obj_extractor := data_processor.feature_extractors["objectives"]) is not None:
        if isinstance(obj_extractor, TabularFeatureExtractor) and obj_extractor.columns:
            objective_values = {k: 0 for k in obj_extractor.columns}

    pd_input = pd.DataFrame.from_dict(
        {
            **{k: [v] * n_items for k, v in (input_parameters or {}).items()},
            **objective_values,
            **input_variables,
            "id": keys,
        }
    )
    pd_input.set_index("id", inplace=True, drop=False)
    iterator = data_processor.make_iterator(pd_input, keys, split="test")
    dataloader = iterator.get_dataloader(batch_size=n_items)
    batch_input, _ = next(iter(dataloader))
    return batch_input.to(device) if device else batch_input, iterator
