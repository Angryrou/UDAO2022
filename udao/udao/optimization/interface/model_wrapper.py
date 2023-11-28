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
    original_input_vars = input_batch.feature_input.clone().detach()
    mask = th.tensor(
        [i in input_variables.keys() for i in udao_shape.feature_input_names]
    )
    grad_indices = th.nonzero(mask, as_tuple=False).squeeze()
    input_vars_subvector = input_batch.feature_input[:, grad_indices]
    input_vars_subvector.requires_grad_(True)
    input_batch.feature_input[:, grad_indices] = input_vars_subvector

    optimizer = optim.Adam([input_vars_subvector], lr=lr)
    output = model(input_batch)
    loss = loss_function(output)

    optimizer.zero_grad()
    loss.backward()
    # input_batch.feature_input.grad *= mask.float()
    optimizer.step()
    input_batch.feature_input[:, grad_indices] = input_vars_subvector

    return input_batch.feature_input - original_input_vars
