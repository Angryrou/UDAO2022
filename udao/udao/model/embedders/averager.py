# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 16/02/2023
import dgl
import torch as th
import torch.nn as nn
from attr import dataclass
from udao.model.embedders.base_embedder import BaseEmbedder, EmbedderParams


@dataclass
class AveragerParams(EmbedderParams):
    name: str = "AVGMLP"


class Averager(BaseEmbedder):
    def __init__(self, net_params: AveragerParams) -> None:
        super().__init__(net_params)

        self.emb = nn.Sequential(
            nn.Linear(self.in_feat_size_op, self.embedding_size), nn.ReLU()
        )

    def _embed(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        h = self.emb(h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        h = self.concatenate_op(g)
        return self.normalize_embedding(self._embed(g, h))
