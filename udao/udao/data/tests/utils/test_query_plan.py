from typing import Any

import pytest

from ...utils.query_plan import extract_query_plan_features
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
    query = query_plan_sample.query_plan
    structure, features = extract_query_plan_features(query)
    assert features.size == expected_op_features.size
    assert features.rows_count == expected_op_features.rows_count
    assert structure.incoming_ids == expected_structure.incoming_ids
    assert structure.outgoing_ids == expected_structure.outgoing_ids
    assert structure.node_id2name == expected_structure.node_id2name


def test_compare_same_struct_is_True(sample_plan_1: QueryPlanElements) -> None:
    structure, features = extract_query_plan_features(sample_plan_1.query_plan)
    assert structure.graph_match(structure) is True


def test_compare_different_structures_is_False(
    sample_plan_1: QueryPlanElements, sample_plan_2: QueryPlanElements
) -> None:
    structure, _ = extract_query_plan_features(sample_plan_1.query_plan)
    structure_2, _ = extract_query_plan_features(sample_plan_2.query_plan)
    assert structure.graph_match(structure_2) is False
