import json
from pathlib import Path
from typing import Tuple

import pytest

from ...utils.query_plan_utils import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    extract_query_plan_features,
    format_size,
)


def get_query_plan_sample(
    json_path: str,
) -> Tuple[str, QueryPlanStructure, QueryPlanOperationFeatures]:
    base_dir = Path(__file__).parent
    with open(base_dir / json_path, "r") as f:
        plan_features = json.load(f)
    incoming_ids = plan_features["incoming_ids"]
    outgoing_ids = plan_features["outgoing_ids"]
    names = plan_features["names"]
    row_counts = [float(v) for v in plan_features["row_counts"]]
    sizes = [format_size(s) for s in plan_features["sizes"]]

    return (
        plan_features["query_plan"],
        QueryPlanStructure(names, incoming_ids, outgoing_ids),
        QueryPlanOperationFeatures(row_counts, sizes),
    )


@pytest.mark.parametrize(
    "query_plan_sample",
    (
        get_query_plan_sample(path)
        for path in ["sample_plan_1.json", "sample_plan_2.json"]
    ),
)
def test_logical_struct(
    query_plan_sample: Tuple[str, QueryPlanStructure, QueryPlanOperationFeatures]
) -> None:
    query, expected_structure, expected_op_features = query_plan_sample

    structure, features = extract_query_plan_features(query)
    assert features.sizes == expected_op_features.sizes
    assert features.rows_counts == expected_op_features.rows_counts
    assert structure.incoming_ids == expected_structure.incoming_ids
    assert structure.outgoing_ids == expected_structure.outgoing_ids
    assert structure.node_id2name == expected_structure.node_id2name


def test_compare_same_struct_is_True() -> None:
    query_plan_sample = get_query_plan_sample("sample_plan_1.json")
    query, expected_structure, expected_op_features = query_plan_sample

    structure, features = extract_query_plan_features(query)
    assert structure.graph_match(structure) is True
    assert structure.nx_graph_match(structure) is True


def test_compare_different_structures_is_False() -> None:
    query_plan_sample_1 = get_query_plan_sample("sample_plan_1.json")
    query_plan_sample_2 = get_query_plan_sample("sample_plan_2.json")
    query, expected_structure, expected_op_features = query_plan_sample_1
    query_2, expected_structure_2, expected_op_features_2 = query_plan_sample_2

    structure, features = extract_query_plan_features(query)
    structure_2, features_2 = extract_query_plan_features(query_2)
    assert structure.graph_match(structure_2) is False
    assert structure.nx_graph_match(structure_2) is False
