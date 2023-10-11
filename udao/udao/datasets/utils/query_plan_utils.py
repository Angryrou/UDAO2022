import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import dgl
import networkx as nx  # brew install graphviz && pip install pydot==1.4.2
from networkx.algorithms import isomorphism


class LogicalOperation:
    """Describes an operation in the logical plan, with associated features."""

    def __init__(self, id: int, value: str) -> None:
        """Generate a logical operation from the logical plan, with its id and text.

        Parameters
        ----------
        id : int
            id of the operation in the query plan (one id per line in the plan)
        value : str
            text of the operation (line in the string representation of the plan)
        """
        self.id: int = id
        self.value: str = value
        self._rank: Optional[float] = None

    id: int
    value: str
    _rank: Optional[float] = None

    @property
    def rank(self) -> float:
        """Compute the rank of the operation,
        i.e. the number of spaces before the operation name."""
        if self._rank is None:
            self._rank = (len(self.value) - len(self.value.lstrip())) / 3
        return self._rank

    @rank.setter
    def rank(self, rank: float) -> None:
        self._rank = rank

    @property
    def link(self) -> str:
        """Compute the link between the current operation and the previous one,
        i.e. the character before the operation name."""
        return self.value.lstrip()[0]

    @property
    def name(self) -> str:
        """Compute the name of the operation: the first word after the link,
        or with additional term after the space in the case of a Relation operation"""
        _, b, c = re.split(r"([a-z])", self.value, 1, flags=re.I)
        name = (b + c).split(" ")[0]
        if name == "Relation":
            name = (b + c).split("[")[0]
        return name

    @property
    def size(self) -> str:
        """Extract the size of the operation, in bytes"""
        return self.value.split("sizeInBytes=")[1].split("B")[0] + "B"

    def get_nrows(self, id_predecessors: List[int], nrows: List[str]) -> str:
        """Extract the number of rows of the operation,
        or infer it from its predecessors if not available"""
        if "rowCount=" in self.value:
            return self.value.split("rowCount=")[1].split(")")[0]
        else:
            if len(id_predecessors) == 1:
                pid = id_predecessors[0]
                assert nrows[pid] is not None
                return nrows[pid]
            elif len(id_predecessors) == 2:
                pid1, pid2 = id_predecessors
                assert nrows[pid1] is not None and nrows[pid2] is not None
                assert float(nrows[pid1]) * float(nrows[pid2]) == 0
                return "0"
            else:
                raise NotImplementedError("More than 2 predecessors")

    def get_unindented(self) -> "LogicalOperation":
        """Return the operation without one level of indentation removed"""
        return LogicalOperation(self.id, self.value.lstrip()[3:])


def _get_tree_structure(
    operations: List[LogicalOperation],
) -> Tuple[List[int], List[int]]:
    """Define the tree structure of the logical plan, as a tuple of two lists,
    where U[i], V[i] are the incoming and outgoing nodes of edge i.

    Parameters
    ----------
    steps : List[LogicalStep]
        List of operation in the logical plan

    Returns
    -------
    Tuple[List[int], List[int]]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    pre_id = operations[0].id
    operations[0].rank = -1
    threads: Dict[float, List[LogicalOperation]] = defaultdict(list)
    for i, op in enumerate(operations):
        if i == 0:
            continue
        if op.link == ":":
            unindented_step = op.get_unindented()
            threads[op.rank].append(unindented_step)

        elif op.link == "+":
            incoming_ids.append(op.id)
            outgoing_ids.append(pre_id)
            pre_id = op.id
        else:
            raise ValueError(i, (op, op.rank, op.link))

    for rank, rank_operations in threads.items():
        incoming_ids.append(rank_operations[0].id)
        outgoing_id_candidates = [
            op.id for op in operations if op.rank == rank - 1 and op.link != ":"
        ]
        if len(outgoing_id_candidates) != 1:
            warnings.warn(
                f"More than one candidate outgoing id for rank {rank}."
                "\n Taking the first candidate."
            )
        outgoing_ids.append(outgoing_id_candidates[0])
        sub_from_ids, sub_to_ids = _get_tree_structure(rank_operations)
        incoming_ids += sub_from_ids
        outgoing_ids += sub_to_ids
    return incoming_ids, outgoing_ids


def extract_operation_features(
    logical_plan: str,
) -> Tuple[List[str], List[str], List[str], List[int], List[int]]:
    """Extract:
    - features of the operations in the logical plan
    - the tree structure of the logical plan
    """
    operations = [
        LogicalOperation(id=i, value=step)
        for i, step in enumerate(logical_plan.splitlines())
    ]
    nops = len(operations)
    nnames: List[str] = [""] * nops
    sizes: List[str] = [""] * nops
    nrows: List[str] = [""] * nops

    incoming_ids, outgoing_ids = _get_tree_structure(operations)
    predecessors: Dict[int, List[int]] = defaultdict(list)
    for f, t in zip(incoming_ids, outgoing_ids):
        predecessors[t].append(f)

    for step in reversed(operations):
        nnames[step.id] = step.name
        sizes[step.id] = step.size
        nrows[step.id] = step.get_nrows(predecessors[step.id], nrows)

    return sizes, nrows, nnames, incoming_ids, outgoing_ids


class SqlStructData:
    def __init__(
        self,
        node_id2name: Dict[int, str],
        from_ids: List[int],
        to_ids: List[int],
        old: Any,
    ):
        self.node_id2name = node_id2name
        self.from_ids = from_ids
        self.to_ids = to_ids
        g = dgl.graph((from_ids, to_ids))
        G = dgl.to_networkx(g)
        nx.set_node_attributes(G, self.node_id2name, name="nname")
        self.g = g
        self.G = G
        self.old = old

    # Think of the structure
    # def plot(self, dir_name, title):
    #    plot_nx_graph(self.G, self.node_id2name, dir_name=dir_name, title=title)

    def graph_match(self, p2: "SqlStructData") -> Optional[Dict[int, int]]:
        G1, G2 = self.G, p2.G
        matcher = isomorphism.GraphMatcher(
            G1, G2, node_match=lambda n1, n2: n1["nname"] == n2["nname"]
        )
        if matcher.match():
            return matcher.mapping  # type: ignore
        else:
            return None


def compute_logical_structure(
    logical_plan: str,
) -> Tuple[SqlStructData, List[str], List[str]]:
    sizes, nrows, nnames, from_ids, to_ids = extract_operation_features(logical_plan)
    struct = SqlStructData(
        node_id2name={i: n for i, n in enumerate(nnames)},
        from_ids=from_ids,
        to_ids=to_ids,
        old=None,
    )
    return struct, sizes, nrows
