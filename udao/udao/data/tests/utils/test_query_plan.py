from typing import Any

import pytest

from ...utils.query_plan import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    compute_meta_features,
    extract_query_plan_features,
)
from ..conftest import QueryPlanElements


# Define a custom fixture to handle both sample plans
@pytest.fixture(params=["sample_plan_1", "sample_plan_2"])
def query_plan_sample(request: Any) -> QueryPlanElements:
    return request.getfixturevalue(request.param)


def test_logical_struct(
    query_plan_sample: QueryPlanElements,
) -> None:
    expected_structure = query_plan_sample.structure
    expected_op_features = query_plan_sample.features
    expected_meta_features = query_plan_sample.meta_features
    query = query_plan_sample.query_plan
    structure, features, meta_features = extract_query_plan_features(query)
    assert features.size == expected_op_features.size
    assert features.rows_count == expected_op_features.rows_count
    assert meta_features["meta_size"] == expected_meta_features["meta_size"]
    assert meta_features["meta_rows_count"] == expected_meta_features["meta_rows_count"]
    assert structure.incoming_ids == expected_structure.incoming_ids
    assert structure.outgoing_ids == expected_structure.outgoing_ids
    assert structure.node_id2name == expected_structure.node_id2name


def test_compare_same_struct_is_True(sample_plan_1: QueryPlanElements) -> None:
    structure, _, _ = extract_query_plan_features(sample_plan_1.query_plan)
    assert structure.graph_match(structure) is True


def test_compare_different_structures_is_False(
    sample_plan_1: QueryPlanElements, sample_plan_2: QueryPlanElements
) -> None:
    structure, _, _ = extract_query_plan_features(sample_plan_1.query_plan)
    structure_2, _, _ = extract_query_plan_features(sample_plan_2.query_plan)
    assert structure.graph_match(structure_2) is False


def test_compute_meta_features_one_node() -> None:
    structure = QueryPlanStructure(["a", "b", "c"], [0, 1], [1, 2])
    features = QueryPlanOperationFeatures([1.0, 2.0, 3.0], [10, 20, 30])
    meta_features = compute_meta_features(structure, features)
    assert meta_features["meta_size"] == 10
    assert meta_features["meta_rows_count"] == 1.0


def test_compute_meta_features_two_nodes() -> None:
    structure = QueryPlanStructure(
        ["a", "b", "c", "d", "e", "f"], [0, 1, 3, 4], [1, 2, 4, 5]
    )
    features = QueryPlanOperationFeatures(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10, 20, 30, 40, 50, 60]
    )
    meta_features = compute_meta_features(structure, features)
    assert meta_features["meta_size"] == 50
    assert meta_features["meta_rows_count"] == 5
