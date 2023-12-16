from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import pandas as pd
import pytorch_warmup as warmup
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError
from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler, DataHandlerParams
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import UdaoModel
from udao.model.module import LearningParams, UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization import concepts
from udao.optimization.soo.mogd import MOGD
from udao.utils.logging import logger

logger.setLevel("INFO")
if __name__ == "__main__":
    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 512

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    #### Data definition ####
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(
                columns=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
                + ["s1", "s2", "s3", "s4"]
                + ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["latency"]),
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
                NormalizePreprocessor(MinMaxScaler(), "graph_meta_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=8))
            ),
        ),
    )

    base_dir = Path(__file__).parent
    lqp_df = pd.read_csv(str(base_dir / "data/LQP.csv"))
    brief_df = pd.read_csv(str(base_dir / "data/brief.csv"))
    cols_to_use = lqp_df.columns.difference(brief_df.columns)

    df = brief_df.merge(
        lqp_df[["id", *cols_to_use]],
        on="id",
    )
    data_handler = DataHandler(
        df,
        DataHandlerParams(
            index_column="id",
            stratify_on="tid",
            dryrun=True,
            data_processor=data_processor,
        ),
    )

    split_iterators = data_handler.get_iterators()
    #### Model definition and training ####

    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": 1024,
            "op_groups": ["cbo", "op_enc", "type"],
            "type_embedding_dim": 8,
            "embedding_normalizer": None,
        },
        regressor_params={"n_layers": 2, "hidden_dim": 32, "dropout": 0.1},
    )
    module = UdaoModule(
        model,
        ["latency"],
        loss=WMAPELoss(),
        learning_params=LearningParams(init_lr=1e-3, min_lr=1e-5, weight_decay=1e-2),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    tb_logger = TensorBoardLogger("tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-val_WMAPE={val_latency_WeightedMeanAbsolutePercentageError:.2f}",
        auto_insert_metric_name=False,
    )
    train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    # split_iterators["train"].set_augmentations(
    #    [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    # )

    scheduler = UdaoLRScheduler(setup_cosine_annealing_lr, warmup.UntunedLinearWarmup)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=2,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(batch_size),
        val_dataloaders=split_iterators["val"].get_dataloader(batch_size),
    )

    #### Optimization
    with open(base_dir / "data" / "sample_query.txt", "r") as f:
        plan = f.read()

    input_parameters = {
        "plan": plan,
        "k1": 6.0,
        "k2": 2.0,
        "k3": 5.0,
        "k4": 1.0,
        "k5": 3.0,
        "k6": 1.0,
        "k7": 1.0,
        "k8": 67.0,
        "s1": 2.0,
        "s2": 3.0,
        "s3": 4.0,
        "s4": 0.0,
        "m1": 0.2562,
        "m2": 0.0362486655254638,
        "m3": 0.1908,
        "m4": 21.956,
        "m5": 26903.4,
        "m6": 122.46,
        "m7": 19789.84,
        "m8": 846.0800000000002,
        "latency": 0,  # dummy
    }

    def n_cores(
        input_variables: concepts.InputVariables,
        input_parameters: concepts.InputParameters = None,
    ) -> th.Tensor:
        return th.tensor((input_variables["k3"]) * input_variables["k1"])

    problem = concepts.MOProblem(
        objectives=[
            concepts.Objective(
                name="latency",
                direction_type="MIN",
                function=concepts.ModelComponent(
                    data_processor=data_processor, model=model
                ),
            ),
            concepts.Objective(
                name="cloud_cost", direction_type="MIN", function=n_cores
            ),
        ],
        variables={
            "k1": concepts.IntegerVariable(2, 16),
            "k2": concepts.IntegerVariable(2, 5),
            "k3": concepts.IntegerVariable(4, 10),
        },
        input_parameters=input_parameters,
        constraints=[],
    )
    mogd = MOGD(
        MOGD.Params(
            learning_rate=0.1,
            weight_decay=0.1,
            max_iters=100,
            patience=10,
            seed=0,
            multistart=10,
            objective_stress=0.1,
            batch_size=10,
        )
    )
    so_problem = concepts.SOProblem(
        objective=problem.objectives[0],
        variables=problem.variables,
        constraints=problem.constraints,
        input_parameters=problem.input_parameters,
    )
    soo_obj, soo_var = mogd.solve(so_problem)
    logger.info(f"Found solution: {soo_obj}, {soo_var}")
    """
    mo_solver = SequentialProgressiveFrontier(
        solver=mogd,
        params=SequentialProgressiveFrontier.Params(),
    )

    mo_solver.solve(problem)
    """
