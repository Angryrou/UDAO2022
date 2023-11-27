from typing import Any, Callable, Dict

import torch as th
import torch.optim as optim

from ...data.handler.data_processor import DataProcessor
from ...model.model import UdaoModel


def gradient_descent(
    data_processor: DataProcessor,
    model: UdaoModel,
    input_non_decision: Dict[str, Any],
    input_variables: Dict[str, list],
    loss_function: Callable,
    lr: float = 1,
) -> th.Tensor:
    """Temporary function that mimics part of what we'll
    do in MOGD (gradient descent but not clipping)"""
    input_batch, udao_shape = data_processor.derive_batch_input(
        input_non_decision, input_variables
    )

    input_vars = input_batch.feature_input.clone().detach()
    input_batch.feature_input.requires_grad = True
    optimizer = optim.Adam([input_batch.feature_input], lr=lr)
    output = model(input_batch)
    loss = loss_function(output)
    mask = th.tensor(
        [
            1 if i in input_variables.keys() else 0
            for i in udao_shape.feature_input_names
        ]
    )
    optimizer.zero_grad()
    loss.backward()
    input_batch.feature_input.grad *= mask.float()
    optimizer.step()
    return input_batch.feature_input - input_vars
