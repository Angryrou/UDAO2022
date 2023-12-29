import torch as th
from params import MySplitIteratorParams, get_graph_avg_params

from model import (
    GraphAverageMLPParams,
    MyLearningParams,
    checkpoint_model_structure,
    get_graph_avg_mlp,
    get_tuned_trainer,
)
from udao.utils.logging import logger
from utils import (
    TABULAR_LQP,
    TABULAR_QS,
    PathWatcher,
    get_split_iterators,
    tensor_dtypes,
)

logger.setLevel("INFO")
if __name__ == "__main__":
    params = get_graph_avg_params()
    device = "gpu" if th.cuda.is_available() else "cpu"
    objectives = ["latency_s", "io_mb"]
    th.set_default_dtype(tensor_dtypes)  # type: ignore

    # Data definition
    tabular_columns = TABULAR_LQP if params.q_type in ["q_compile", "q"] else TABULAR_QS
    split_params = MySplitIteratorParams(
        op_groups=params.op_groups,
        tabular_columns=tabular_columns,
        objectives=objectives,
        lpe_size=params.lpe_size,
        vec_size=params.vec_size,
        seed=params.seed,
        q_type=params.q_type,
        debug=params.debug,
    )
    pw = PathWatcher(params.benchmark, params.debug, split_params.hash())
    split_iterators = get_split_iterators(pw=pw, params=split_params)
    # Model definition and training
    model_params = GraphAverageMLPParams(
        iterator_shape=split_iterators["train"].shape,
        output_size=params.output_size,
        op_groups=params.op_groups,
        type_embedding_dim=params.type_embedding_dim,
        embedding_normalizer=params.embedding_normalizer,
        n_layers=params.n_layers,
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
    )
    learning_params = MyLearningParams(
        epochs=params.epochs,
        batch_size=params.batch_size,
        init_lr=params.init_lr,
        min_lr=params.min_lr,
        weight_decay=params.weight_decay,
    )
    model = get_graph_avg_mlp(model_params)
    # prepare the model structure path
    ckp_header = checkpoint_model_structure(pw=pw, model_params=model_params)
    trainer, module = get_tuned_trainer(
        ckp_header,
        model,
        split_iterators,
        objectives,
        learning_params,
        device,
        num_workers=0 if params.debug else params.num_workers,
    )
    test_results = trainer.test(
        model=module,
        dataloaders=split_iterators["test"].get_dataloader(
            batch_size=params.batch_size,
            num_workers=0 if params.debug else params.num_workers,
            shuffle=False,
        ),
    )
    print(test_results)
