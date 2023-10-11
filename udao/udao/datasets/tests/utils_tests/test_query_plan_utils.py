from pathlib import Path
from typing import List, Tuple

import dgl
import networkx as nx
import pytest

from udao.udao.datasets.utils.query_plan_utils import compute_logical_structure


@pytest.fixture
def real_query_plan_sample() -> Tuple[str, nx.Graph, List[str], List[str]]:
    base_dir = Path(__file__).parent
    with open(base_dir / "sample_plan.txt", "r") as f:
        query = f.read().replace("\\n", "\n")
    from_ids = [
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
    to_ids = [0, 1, 2, 3, 4, 5, 19, 20, 5, 6, 7, 16, 17, 7, 8, 9, 13, 14, 9, 10, 11]
    node_id2name = {
        0: "GlobalLimit",
        1: "LocalLimit",
        2: "Sort",
        3: "Aggregate",
        4: "Project",
        5: "Join",
        6: "Project",
        7: "Join",
        8: "Project",
        9: "Join",
        10: "Project",
        11: "Filter",
        12: "Relation tpch_100.customer",
        13: "Project",
        14: "Filter",
        15: "Relation tpch_100.orders",
        16: "Project",
        17: "Filter",
        18: "Relation tpch_100.nation",
        19: "Project",
        20: "Filter",
        21: "Relation tpch_100.lineitem",
    }
    nrows = [
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
    sizes = [
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
    graph = dgl.graph((from_ids, to_ids))
    nx_graph = dgl.to_networkx(graph)

    nx.set_node_attributes(nx_graph, node_id2name, name="nname")

    return query, nx_graph, sizes, nrows


def test_logical_struct(
    real_query_plan_sample: Tuple[str, nx.Graph, List[str], List[str]]
) -> None:
    query, expected_nx_graph, expected_sizes, expected_nrows = real_query_plan_sample

    struct, sizes, nrows = compute_logical_structure(query)
    assert sizes == expected_sizes
    assert nrows == expected_nrows
    assert nx.utils.graphs_equal(struct.G, expected_nx_graph) is True  # type: ignore
