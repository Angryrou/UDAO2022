import json
from pathlib import Path
from typing import NamedTuple

import pytest
from udao.data.utils.query_plan import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    format_size,
)

QueryPlanElements = NamedTuple(
    "QueryPlanElements",
    [
        ("query_plan", str),
        ("structure", QueryPlanStructure),
        ("features", QueryPlanOperationFeatures),
    ],
)


def get_query_plan_sample(
    json_path: str,
) -> QueryPlanElements:
    base_dir = Path(__file__).parent
    with open(base_dir / json_path, "r") as f:
        plan_features = json.load(f)
    incoming_ids = plan_features["incoming_ids"]
    outgoing_ids = plan_features["outgoing_ids"]
    names = plan_features["names"]
    row_counts = [float(v) for v in plan_features["row_counts"]]
    sizes = [format_size(s) for s in plan_features["sizes"]]

    return QueryPlanElements(
        plan_features["query_plan"],
        QueryPlanStructure(names, incoming_ids, outgoing_ids),
        QueryPlanOperationFeatures(row_counts, sizes),
    )


@pytest.fixture(scope="session")
def sample_plan_1() -> QueryPlanElements:
    base_dir = Path(__file__).parent
    return get_query_plan_sample(str(base_dir / "assets/sample_plan_1.json"))


@pytest.fixture(scope="session")
def sample_plan_2() -> QueryPlanElements:
    base_dir = Path(__file__).parent
    return get_query_plan_sample(str(base_dir / "assets/sample_plan_2.json"))
