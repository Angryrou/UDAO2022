import pytest
import torch as th

from ...regressors.mlp import MLP, MLPParams


@pytest.mark.parametrize("embed_dim, feat_dim", [(10, 10), (5, 5)])
def test_udao_mlp_forward_shape(embed_dim: int, feat_dim: int) -> None:
    sample_mlp_params = MLPParams(
        input_embedding_dim=embed_dim,
        input_features_dim=feat_dim,
        output_dim=64,
        n_layers=3,
        hidden_dim=128,
        dropout=0.5,
    )
    model = MLP(sample_mlp_params)

    sample_embedding = th.rand((32, embed_dim))
    sample_inst_feat = th.rand((32, feat_dim))

    output = model.forward(sample_embedding, sample_inst_feat)

    assert output.shape == (32, 64)
