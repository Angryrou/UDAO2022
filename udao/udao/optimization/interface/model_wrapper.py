from typing import Any, Callable, Dict

import pandas as pd
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from ...data.handler.data_processor import DataProcessor
from ...data.iterators.base_iterator import UdaoIterator
from ...model.model import UdaoModel


class ModelWrapper:
    def __init__(self, model: UdaoModel, data_processor: DataProcessor) -> None:
        self.model = model
        self.data_processor = data_processor

    def compute_iterator(
        self,
        input_non_decision: Dict[str, Any],
        input_variables: Dict[str, list],
        n_vars: int,
    ) -> UdaoIterator:
        keys = [f"{i}" for i in range(n_vars)]
        pd_input = pd.DataFrame.from_dict(
            {
                **{k: [v] * n_vars for k, v in input_non_decision.items()},
                **input_variables,
                "id": keys,
            }
        )
        pd_input.set_index("id", inplace=True)
        iterator = self.data_processor.make_iterator(pd_input, keys, split="test")
        return iterator  # type: ignore


def gradient_descent(
    model_wrapper: ModelWrapper,
    input_non_decision: Dict[str, Any],
    input_variables: Dict[str, list],
    n_vars: int,
    loss_function: Callable,
    lr: float = 1,
) -> th.Tensor:
    """Temporary function that mimicks part of what we'll
    do in MOGD (gradient descent but not clipping)"""
    iterator = model_wrapper.compute_iterator(
        input_non_decision, input_variables, n_vars
    )
    iterator.get_dataloader(n_vars)
    dl = DataLoader(iterator, batch_size=2, collate_fn=iterator.collate)
    input_batch, _ = next(iter(dl))
    # input_batch, _ = next(iter(dataloader))
    input_vars = input_batch.feature_input.clone().detach()
    input_batch.feature_input.requires_grad = True
    optimizer = optim.Adam([input_batch.feature_input], lr=lr)
    udao_shape = iterator.get_iterator_shape()
    udao_shape.feature_input_names
    output = model_wrapper.model(input_batch)
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
