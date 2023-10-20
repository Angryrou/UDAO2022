import dgl
import numpy as np
import pandas as pd
import pytest
import torch as th
from udao.data.containers import DataFrameContainer, QueryStructureContainer
from udao.data.iterators import QueryPlanIterator
from udao.data.utils.query_plan import QueryPlanStructure


@pytest.fixture
def sample_iterator() -> QueryPlanIterator:
    """Builds a sample QueryPlanIterator, with 3 graphs from 2 templates."""
    keys = ["a", "b", "c"]
    arrays = [["a", "a", "b", "b", "b", "c", "c"], [0, 1, 0, 1, 2, 0, 1]]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=("plan_id", "operation_id"))

    graph_features = pd.DataFrame(
        data=np.vstack([np.linspace(i, i + 1, 2) for i in range(7)]),
        index=multi_index,
        columns=["rows_count", "size"],
    )
    embeddings_features = pd.DataFrame(
        data=np.vstack([np.linspace(i, i + 1, 10) for i in range(7)]),
        index=multi_index,
        columns=[f"emb_{i}" for i in range(10)],
    )
    key_to_template = {"a": 1, "b": 2, "c": 1}

    template_plans = {
        1: QueryPlanStructure(["node_1", "node_2"], [0], [1]),
        2: QueryPlanStructure(["node_1", "node_2", "node_3"], [0, 1], [1, 2]),
    }
    structure_container = QueryStructureContainer(
        graph_features, template_plans, key_to_template
    )
    return QueryPlanIterator(
        keys, structure_container, DataFrameContainer(embeddings_features)
    )


class TestQueryPlanIterator:
    def test_len(self, sample_iterator: QueryPlanIterator) -> None:
        assert len(sample_iterator) == len(sample_iterator.keys)
        assert len(sample_iterator) == len(
            sample_iterator.query_structure_container.key_to_template
        )

    def test_get_item(self, sample_iterator: QueryPlanIterator) -> None:
        first_sample = sample_iterator[0]
        a_sample = sample_iterator._get_graph("a")
        for key in first_sample.ndata:
            assert th.equal(first_sample.ndata[key], a_sample.ndata[key])  # type: ignore

    def test_get_graph(self, sample_iterator: QueryPlanIterator) -> None:
        graph = sample_iterator._get_graph("a")
        assert isinstance(graph, dgl.DGLGraph)
        assert graph.number_of_nodes() == 2
        assert th.equal(
            graph.ndata["cbo"], th.tensor([np.linspace(i, i + 1, 2) for i in range(2)])  # type: ignore
        )
        assert th.equal(
            graph.ndata["op_encs"],  # type: ignore
            th.tensor([np.linspace(i, i + 1, 10) for i in range(2)]),
        )
