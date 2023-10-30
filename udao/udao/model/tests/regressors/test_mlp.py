import pytest
import torch as th

from ...regressors.mlp import MLP, MLPParams
from ...utils import set_deterministic_torch


@pytest.mark.parametrize(
    "embed_dim, feat_dim, expected_output",
    [
        (
            10,
            10,
            th.tensor(
                [
                    [1.427368e00, 1.429442e00, 7.568769e-01, 1.026550e00, 7.447854e-01],
                    [1.426292e00, 1.357975e00, 7.294571e-01, 1.020146e00, 7.296020e-01],
                ]
            ),
        ),
        (
            5,
            5,
            th.tensor(
                [
                    [1.392054e00, 1.645180e00, 1.155030e00, 1.078174e00, 7.341877e-01],
                    [1.543798e00, 1.272568e00, 1.473847e00, 9.997230e-01, 6.884866e-01],
                ]
            ),
        ),
    ],
)
def test_udao_mlp_forward_shape(
    embed_dim: int, feat_dim: int, expected_output: th.Tensor
) -> None:
    set_deterministic_torch(0)
    sample_mlp_params = MLPParams(
        input_embedding_dim=embed_dim,
        input_features_dim=feat_dim,
        output_dim=5,
        n_layers=3,
        hidden_dim=5,
        dropout=0.5,
    )
    model = MLP(sample_mlp_params)
    sample_embedding = th.rand((2, embed_dim))
    sample_inst_feat = th.rand((2, feat_dim))
    output = model.forward(sample_embedding, sample_inst_feat)
    assert output.shape == (2, 5)
    assert th.allclose(output, expected_output, rtol=1e-5)
