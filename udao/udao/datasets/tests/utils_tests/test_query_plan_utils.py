from pathlib import Path
from typing import Tuple

import pytest

from ...utils.query_plan_utils import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    extract_query_plan_features,
    format_size,
)


@pytest.fixture
def real_query_plan_sample() -> (
    Tuple[str, QueryPlanStructure, QueryPlanOperationFeatures]
):
    base_dir = Path(__file__).parent
    with open(base_dir / "sample_plan.txt", "r") as f:
        query = f.read().replace("\\n", "\n")
    incoming_ids = [
        1,
        2,
        3,
        4,
        5,
        19,
        20,
        21,
        6,
        7,
        16,
        17,
        18,
        8,
        9,
        13,
        14,
        15,
        10,
        11,
        12,
    ]
    outgoing_ids = [
        0,
        1,
        2,
        3,
        4,
        5,
        19,
        20,
        5,
        6,
        7,
        16,
        17,
        7,
        8,
        9,
        13,
        14,
        9,
        10,
        11,
    ]
    names = [
        "GlobalLimit",
        "LocalLimit",
        "Sort",
        "Aggregate",
        "Project",
        "Join",
        "Project",
        "Join",
        "Project",
        "Join",
        "Project",
        "Filter",
        "Relation tpch_100.customer",
        "Project",
        "Filter",
        "Relation tpch_100.orders",
        "Project",
        "Filter",
        "Relation tpch_100.nation",
        "Project",
        "Filter",
        "Relation tpch_100.lineitem",
    ]
    rows_counts = [
        float(v)
        for v in [
            "20",
            "2.42E+7",
            "2.42E+7",
            "2.42E+7",
            "2.42E+7",
            "2.42E+7",
            "6.16E+6",
            "6.16E+6",
            "6.16E+6",
            "6.16E+6",
            "1.50E+7",
            "1.50E+7",
            "1.50E+7",
            "5.68E+6",
            "5.68E+6",
            "5.74E+6",
            "25",
            "25",
            "25",
            "2.00E+8",
            "2.00E+8",
            "6.00E+8",
        ]
    ]
    sizes = [
        format_size(s)
        for s in [
            "2.7 KiB",
            "5.4 GiB",
            "5.4 GiB",
            "5.4 GiB",
            "5.4 GiB",
            "5.8 GiB",
            "1357.0 MiB",
            "1451.0 MiB",
            "1286.5 MiB",
            "1333.5 MiB",
            "2.9 GiB",
            "3.3 GiB",
            "3.3 GiB",
            "130.0 MiB",
            "877.4 MiB",
            "886.5 MiB",
            "900.0 B",
            "3.2 KiB",
            "3.2 KiB",
            "6.0 GiB",
            "34.6 GiB",
            "103.9 GiB",
        ]
    ]

    return (
        query,
        QueryPlanStructure(names, incoming_ids, outgoing_ids),
        QueryPlanOperationFeatures(rows_counts, sizes),
    )


def test_logical_struct(
    real_query_plan_sample: Tuple[str, QueryPlanStructure, QueryPlanOperationFeatures]
) -> None:
    query, expected_structure, expected_op_features = real_query_plan_sample

    structure, features = extract_query_plan_features(query)
    assert features.sizes == expected_op_features.sizes
    assert features.rows_counts == expected_op_features.rows_counts
    assert structure.incoming_ids == expected_structure.incoming_ids
    assert structure.outgoing_ids == expected_structure.outgoing_ids
    assert structure.node_id2name == expected_structure.node_id2name
