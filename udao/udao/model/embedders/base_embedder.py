from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import dgl
import torch as th
import torch.nn as nn

from .layers.iso_bn import IsoBN

NormalizerType = Literal["BN", "LN", "IsoBN"]


@dataclass
class EmbedderParams:
    input_size: int  # depends on the data
    """The size of the input features."""
    output_size: int
    """The size of the output embedding."""
    op_groups: Sequence[str]
    """The groups of operation features to be included in the embedding."""
    type_embedding_dim: int
    """The dimension of the operation type embedding."""
    embedding_normalizer: Optional[NormalizerType]
    """Name of the normalizer to use for the output embedding."""
    n_op_types: int  # depends on the data
    """The number of operation types - defines the
    size of the operation type embedding."""


class BaseEmbedder(nn.Module):
    """Base class for Embedder networks.
    Takes care of preparing the input features for the
    embedding layer, and normalizing the output embedding.

    Parameters
    ----------
    net_params : EmbedderParams
    """

    def __init__(self, net_params: EmbedderParams) -> None:
        super().__init__()
        self.input_size = net_params.input_size
        self.embedding_size = net_params.output_size

        op_groups = net_params.op_groups
        self.op_type = "ch1_type" in op_groups
        self.op_cbo = "ch1_cbo" in op_groups
        self.op_enc = "ch1_enc" in op_groups
        if self.op_type:
            self.op_embedder = nn.Embedding(
                net_params.n_op_types, net_params.type_embedding_dim
            )
        self.out_norm: Optional[nn.Module] = None
        if net_params.embedding_normalizer is None:
            self.out_norm = None
        elif net_params.embedding_normalizer == "BN":
            self.out_norm = nn.BatchNorm1d(self.embedding_size)
        elif net_params.embedding_normalizer == "LN":
            self.out_norm = nn.LayerNorm(self.embedding_size)
        elif net_params.embedding_normalizer == "IsoBN":
            self.out_norm = IsoBN(self.embedding_size)
        else:
            raise ValueError(net_params.embedding_normalizer)

    def concatenate_op_features(self, g: dgl.DGLGraph) -> th.Tensor:
        """Concatenate the operation features into a single tensor.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph

        Returns
        -------
        th.Tensor
            output tensor of shape (num_nodes, input_size)
        """
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["enc"])
        return th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]

    def normalize_embedding(self, embedding: th.Tensor) -> th.Tensor:
        """Normalizes the embedding."""
        if self.out_norm is not None:
            embedding = self.out_norm(embedding)
        return embedding
