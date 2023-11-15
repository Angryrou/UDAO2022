from dataclasses import dataclass
from typing import Dict, Tuple

import dgl
import numpy as np
import pandas as pd

from ...data.containers.base_container import BaseContainer
from ..utils.query_plan import QueryPlanStructure


@dataclass
class QueryStructureContainer(BaseContainer):
    """Container for the query structure and features of a query plan."""

    graph_features: pd.DataFrame
    """ Stores the features of the operations in the query plan."""
    graph_meta_features: pd.DataFrame
    """ Stores the meta features of the operations in the query plan."""
    template_plans: Dict[int, QueryPlanStructure]
    """Link a template id to a QueryPlanStructure"""
    key_to_template: Dict[str, int]
    """Link a key to a template id."""

    def get(self, key: str) -> Tuple[dgl.DGLGraph, np.ndarray, np.ndarray]:
        graph_features = self.graph_features.loc[key].values
        graph_meta_features = self.graph_meta_features.loc[key].values
        template_id = self.key_to_template[key]
        template_graph = self.template_plans[template_id].graph

        return template_graph, graph_features, graph_meta_features  # type: ignore
