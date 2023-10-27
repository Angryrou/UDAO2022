from typing import Tuple

import dgl
import pytest
import torch

from ....embedders.layers.multi_head_attention import (
    MultiHeadAttentionLayer,
    QFMultiHeadAttentionLayer,
    RAALMultiHeadAttentionLayer,
)

FixtureType = Tuple[MultiHeadAttentionLayer, dgl.DGLGraph, torch.Tensor]


@pytest.fixture
def mha_fixture() -> FixtureType:
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    h = torch.rand((4, 5))
    mha = MultiHeadAttentionLayer(in_dim=5, out_dim=3, n_heads=2, use_bias=True)

    return mha, g, h


class TestMultiHeadAttentionLayer:
    def test_init(self) -> None:
        mha = MultiHeadAttentionLayer(in_dim=5, out_dim=3, n_heads=2, use_bias=True)
        assert mha.Q.in_features == 5
        assert mha.Q.out_features == 6
        assert mha.K.in_features == 5
        assert mha.K.out_features == 6
        assert mha.V.in_features == 5
        assert mha.V.out_features == 6

    def test_forward(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        out = mha(g, h)
        assert out.size() == (4, 2, 3)

    def test_compute_query_key_value(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        g = mha.compute_query_key_value(g, h)
        for key in ["Q_h", "K_h", "V_h"]:
            assert key in g.ndata
            assert g.ndata[key].size() == (4, 2, 3)  # type: ignore
        pass

    def test_compute_attention(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        graph = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(graph)
        assert "score" in g.edata

    def test_attention_has_effect(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        out = mha(g, h)
        original_h = h.clone()
        assert not torch.equal(original_h, out[:, 0, :])


class TestQFMultiHeadAttentionLayer:
    def test_compute_attention(self) -> None:
        g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        h = torch.rand((4, 5))
        g.edata["dist"] = torch.tensor([1, 2, 3])
        attention_bias = torch.tensor([0.1, 0.2, 0.3])
        mha = QFMultiHeadAttentionLayer(
            in_dim=5,
            out_dim=3,
            n_heads=2,
            use_bias=True,
            attention_bias=attention_bias,
        )
        g = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(g)
        assert "score" in g.edata


class TestRAALMultiHeadAttentionLayer:
    def test_compute_attention(self) -> None:
        g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        g.ndata["sid"] = torch.tensor([0, 0, 0, 0])
        non_siblings_map = {0: {0: [2, 3], 1: [0, 3], 2: [0, 1]}}
        h = torch.rand((4, 5))
        mha = RAALMultiHeadAttentionLayer(
            in_dim=5,
            out_dim=3,
            n_heads=2,
            use_bias=True,
            non_siblings_map=non_siblings_map,
        )
        g = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(g)
        assert "score" in g.edata
