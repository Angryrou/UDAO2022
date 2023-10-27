from typing import List, Optional

import pytest
import torch as th

from ....regressors.layers.mlp_readout import MLPReadout


@pytest.mark.parametrize("agg_dims", [None, [10, 15], [12]])
def test_initialize_with_aggregation_layers(agg_dims: Optional[List[int]]) -> None:
    model = MLPReadout(10, 20, 5, dropout=0, n_layers=3, agg_dims=agg_dims)

    total_layers = 3 + 1  # inner layers + output layer
    if agg_dims:
        total_layers += len(agg_dims)

    assert len(model.FC_layers) == total_layers
    assert model.BN_layers is not None
    assert len(model.BN_layers) == total_layers - 1


def test_init_with_dropout() -> None:
    model = MLPReadout(10, 20, 5, dropout=0.5, n_layers=3, agg_dims=[10, 15])
    assert model.BN_layers is None


@pytest.mark.parametrize("agg_dims", [None, [10, 15], [12]])
@pytest.mark.parametrize("dropout", [0, 0.5])
def test_mlp_readout_forward(agg_dims: Optional[List[int]], dropout: float) -> None:
    model = MLPReadout(10, 20, 5, dropout=dropout, n_layers=3, agg_dims=agg_dims)
    input_tensor = th.rand((32, 10))
    output = model.forward(input_tensor)
    assert output.shape == (32, 5)
