# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 16/02/2023
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import dgl
import torch as th
import torch.nn as nn

from .layers.iso_bn import IsoBN


@dataclass
class EmbedderParams:
    in_feat_size_op: int  # depends on the data
    embedding_size: int
    op_groups: Sequence[str]
    ch1_type_dim: int
    embedding_normalizer: str
    n_op_types: int  # depends on the data
    name: str


class BaseEmbedder(nn.Module):
    def __init__(self, net_params: EmbedderParams) -> None:
        super().__init__()
        self.name = net_params.name
        self.in_feat_size_op = net_params.in_feat_size_op  # depends on the data
        self.embedding_size = net_params.embedding_size

        op_groups = net_params.op_groups
        self.op_type = "ch1_type" in op_groups
        self.op_cbo = "ch1_cbo" in op_groups
        self.op_enc = "ch1_enc" in op_groups
        if self.op_type:
            self.op_embedder = nn.Embedding(
                net_params.n_op_types, net_params.ch1_type_dim
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

    def concatenate_op(self, g: dgl.DGLGraph) -> th.Tensor:
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["enc"])
        return th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]

    def normalize_embedding(self, embedding: th.Tensor) -> th.Tensor:
        if self.out_norm is not None:
            embedding = self.out_norm(embedding)
        return embedding

    @abstractmethod
    def forward(self):  # type: ignore
        ...
