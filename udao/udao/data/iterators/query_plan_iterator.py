from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import dgl
import torch as th

from ...data.containers.tabular_container import TabularContainer
from ...utils.interfaces import UdaoInput, UdaoInputShape
from ..containers import QueryStructureContainer
from .base_iterator import UdaoIterator


@dataclass
class QueryPlanInput(UdaoInput[dgl.DGLGraph]):
    """The embedding input is a dgl.DGLGraph"""

    pass


class QueryPlanIterator(UdaoIterator[QueryPlanInput, UdaoInputShape]):
    """
    Iterator that returns a dgl.DGLGraph for each key, with associated node features.
    The features are stored in the graph.ndata dictionary.
    The features are expected to be float tensors, and to be of the same length
    as the number of nodes in the graph.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    tabular_features : TabularContainer
        Container for the tabular features associated with the plan
    objectives : TabularContainer
        Container for the objectives associated with the plan
    query_structure : QueryStructureContainer
        Wrapper around the graph structure and the features for each query plan
    kwargs: BaseContainer
        Variable number of other features to add to the graph, e.g. embeddings
    """

    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
        query_structure: QueryStructureContainer,
        **kwargs: TabularContainer,
    ):
        super().__init__(
            keys=keys,
            tabular_features=tabular_features,
            objectives=objectives,
        )
        self.query_structure_container = query_structure
        self.base_graph_features = ["cbo"]
        self.other_graph_features = kwargs

    def __len__(self) -> int:
        return len(self.keys)

    def _get_graph_and_meta(self, key: str) -> Tuple[dgl.DGLGraph, th.Tensor]:
        """Returns the graph corresponding to the key,
        associated with features as th.tensor,
        and the meta information.
        """
        query = self.query_structure_container.get(key)
        graph = query.template_graph
        graph.ndata["cbo"] = th.tensor(query.graph_features, dtype=self.tensors_dtype)
        graph.ndata["op_gid"] = th.tensor(query.operation_types, dtype=th.int32)
        for feature, container in self.other_graph_features.items():
            graph.ndata[feature] = th.tensor(
                container.get(key), dtype=self.tensors_dtype
            )
        return graph, th.tensor(query.meta_features, dtype=self.tensors_dtype)

    def _getitem(self, idx: int) -> Tuple[QueryPlanInput, th.Tensor]:
        key = self.keys[idx]
        features = th.tensor(self.tabular_features.get(key), dtype=self.tensors_dtype)
        objectives = th.tensor(self.objectives.get(key), dtype=self.tensors_dtype)
        graph, meta_input = self._get_graph_and_meta(key)
        features = th.cat([meta_input, features])
        input_data = QueryPlanInput(graph, features)
        return input_data, objectives

    @property
    def shape(self) -> UdaoInputShape[Dict[str, int]]:
        """Returns the dimensions of the iterator inputs and outputs."""

        sample_input, sample_output = self._get_sample()
        embedding_input_shape = {}
        graph_feature_names = self.base_graph_features + list(
            self.other_graph_features.keys()
        )
        for feature_name in graph_feature_names:
            embedding_input_shape[feature_name] = sample_input.embedding_input.ndata[
                feature_name
            ].shape[  # type: ignore
                1
            ]
        embedding_input_shape["type"] = len(
            self.query_structure_container.operation_types.unique()
        )
        feature_names = [
            *self.tabular_features.data.columns,
            *self.query_structure_container.graph_meta_features.columns,
        ]
        return UdaoInputShape(
            embedding_input_shape=embedding_input_shape,
            feature_input_names=feature_names,
            output_shape=sample_output.shape[0],
        )

    @staticmethod
    def collate(
        items: List[Tuple[QueryPlanInput, th.Tensor]],
    ) -> Tuple[QueryPlanInput, th.Tensor]:
        """Collate a list of FeatureItem into a single graph."""
        graphs = [item[0].embedding_input for item in items]
        features = th.vstack([item[0].feature_input for item in items])
        objectives = th.vstack([item[1] for item in items])
        return QueryPlanInput(dgl.batch(graphs), features), objectives

    @staticmethod
    def make_graph_augmentation(
        augmentation: Callable[[dgl.DGLGraph], dgl.DGLGraph],
    ) -> Callable:
        def graph_augmentation(
            item: Tuple[QueryPlanInput, th.Tensor]
        ) -> Tuple[QueryPlanInput, th.Tensor]:
            query_plan, output = item
            query_plan.embedding_input = augmentation(query_plan.embedding_input)
            return query_plan, output

        return graph_augmentation
