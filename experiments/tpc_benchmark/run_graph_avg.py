import torch as th
from params import get_graph_avg_params

from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import UdaoModel
from udao.model.regressors.mlp import MLP
from udao.utils.logging import logger
from utils import get_tuned_trainer, magic_extract_iterators

logger.setLevel("INFO")
if __name__ == "__main__":
    params = get_graph_avg_params()
    device = "gpu" if th.cuda.is_available() else "cpu"
    model_sign = "graph_avg"
    objectives = ["latency_s", "io_mb"]

    # Data definition
    split_iterators = magic_extract_iterators(params, objectives)

    # Model definition and training
    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": params.output_size,  # 32
            "op_groups": params.op_groups,  # ["type", "cbo", "op_enc"]
            "type_embedding_dim": params.type_embedding_dim,  # 8
            "embedding_normalizer": params.embedding_normalizer,  # None
        },
        regressor_params={
            "n_layers": params.n_layers,  # 3
            "hidden_dim": params.hidden_dim,  # 512
            "dropout": params.dropout,  # 0.1
        },
    )
    trainer, module = get_tuned_trainer(
        model, split_iterators, objectives, params, model_sign, device
    )

    test_results = trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            params.batch_size,
            num_workers=0 if params.debug else params.num_workers,
            shuffle=False,
        ),
    )
    print(test_results)
