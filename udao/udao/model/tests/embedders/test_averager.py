import dgl
import pytest

from ...embedders.averager import Averager, AveragerParams
from .conftest import generate_dgl_graph


@pytest.fixture
def params_fixture() -> AveragerParams:
    return AveragerParams(
        input_size=7,
        output_size=10,
        op_groups=["ch1_type", "ch1_cbo"],
        type_embedding_dim=5,
        embedding_normalizer=None,
        n_op_types=3,
    )


class TestAverager:
    def test_forward_shape(self, params_fixture: AveragerParams) -> None:
        averager = Averager(params_fixture)
        features_dict = {
            "op_gid": {"size": 1, "type": "int"},
            "cbo": {"size": 2, "type": "float"},
        }
        g1 = generate_dgl_graph(3, 2, features_dict)
        g2 = generate_dgl_graph(5, 4, features_dict)
        g_batch = dgl.batch([g1, g2])
        embedding = averager.forward(g_batch)
        assert embedding.shape == (2, 10)
