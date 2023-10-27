from dataclasses import dataclass
from typing import List, Optional

import torch as th

from .base_regressor import BaseRegressor
from .layers.mlp_readout import MLPReadout


@dataclass
class MLPParams:
    input_embedding_dim: int
    """Size of the embedding part of the input."""
    input_features_dim: int  # depends on the data
    """Size of the tabular features."""
    output_dim: int
    """Size of the output tensor."""
    n_layers: int
    """Number of layers in the MLP"""
    hidden_dim: int
    """Size of the hidden layers outputs."""
    dropout: float
    """Probability of dropout."""
    agg_dims: Optional[List[int]] = None
    """Dimensions of the aggregation layers in the MLP."""


class MLP(BaseRegressor):
    """MLP to compute the final regression from
    the embedding and the tabular input features.
    Parameters
    ----------
    net_params : MLPParams
        For the parameters, see the MLPParams dataclass.
    """

    def __init__(self, net_params: MLPParams) -> None:
        """_summary_"""
        super().__init__()
        self.name = "AVGMLP"
        input_dim = net_params.input_embedding_dim + net_params.input_features_dim

        self.MLP_layers = MLPReadout(
            input_dim=input_dim,
            hidden_dim=net_params.hidden_dim,
            output_dim=net_params.output_dim,
            n_layers=net_params.n_layers,
            dropout=net_params.dropout,
            agg_dims=net_params.agg_dims,
        )

    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        hgi = th.cat([embedding, inst_feat], dim=1)
        out = self.MLP_layers.forward(hgi)
        return th.exp(out)
