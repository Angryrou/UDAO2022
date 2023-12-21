from typing import Dict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytorch_warmup as warmup
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError

from udao.data.containers.tabular_container import TabularContainer
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler, DataHandlerParams
from udao.data.handler.data_processor import create_data_processor, FeaturePipeline
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from udao.data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from udao.model.module import LearningParams, UdaoModule
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.utils import set_deterministic_torch
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization.concepts import Constraint, FloatVariable, Objective, Variable
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.concepts.utils import InputParameters, InputVariables
from udao.optimization.moo.weighted_sum import WeightedSum
from udao.optimization.moo.progressive_frontier import SequentialProgressiveFrontier
from udao.optimization.moo.progressive_frontier import ParallelProgressiveFrontier
from udao.optimization.soo.grid_search_solver import GridSearch
from udao.optimization.soo.mogd import MOGD
from udao.utils.interfaces import UdaoInput
from udao.utils.logging import logger

logger.setLevel("INFO")

#     An example: 2D
#     https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization_problems
#     Binh and Korn function:
#     # minimize:
#     #          f1(x1, x2) = 4 * x_1 * x_1 + 4 * x_2 * x_2
#     #          f2(x1, x2) = (x_1 - 5) * (x_1 - 5) + (x_2 - 5) * (x_2 - 5)
#     # subject to:
#     #          g1(x_1, x_2) = (x_1 - 5) * (x_1 - 5) + x_2 * x_2 <= 25
#     #          g2(x_1, x_2) = (x_1 - 8) * (x_1 - 8) + (x_2 + 3) * (x_2 + 3) >= 7.7
#     #          x_1 in [0, 5], x_2 in [0, 3]
#     """

def set_nn():

    def get_raw_data():
        n = 1000
        np.random.seed(0)
        rand_x_1 = np.random.uniform(0, 5, n)
        rand_x_2 = np.random.uniform(0, 3, n)

        obj1 = 4 * rand_x_1 ** 2 + 4 * rand_x_2 ** 2
        obj2 = (rand_x_1 - 5) ** 2 + (rand_x_2 - 5) ** 2
        const1 = (rand_x_1 - 5) ** 2 + rand_x_2 ** 2
        const2 = (rand_x_1 - 8) ** 2 + (rand_x_2 + 3) ** 2

        avail_inds1 = np.where(const1 <= 25)
        avail_inds2 = np.where(const2 >= 7.7)
        avail_inds = np.intersect1d(avail_inds1, avail_inds2)

        ids = np.arange(1, n + 1, 1)
        df = pd.DataFrame.from_dict({"id": ids[avail_inds],
                                     "v1": rand_x_1[avail_inds], "v2": rand_x_2[avail_inds],
                                     "obj1": obj1[avail_inds], "obj2": obj2[avail_inds],
                                     "const1": const1[avail_inds], "const2": const2[avail_inds]})

        return df

    tensor_dtypes = th.float32
    device = "gpu" if th.cuda.is_available() else "cpu"
    batch_size = 3

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    # Create the dynamic DataHandlerParams class
    # DummyFeatureIterator
    data_processor_getter = create_data_processor(DummyUdaoIterator)

    scaler = MinMaxScaler()
    # Instantiate the dynamic class
    nn_data_processor = data_processor_getter(
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=["v1", "v2"]),
            preprocessors=[NormalizePreprocessor(scaler)],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["obj1", "obj2"])
        )
    )

    df = get_raw_data()
    data_handler = DataHandler(
        df,
        DataHandlerParams(
            index_column="id",
            # stratify_on="obj1",  # stratification is made for discrete classes, it won't work with a continuous value
            dryrun=False,
            data_processor=nn_data_processor,
        ),
    )
    split_iterators = data_handler.get_iterators()
    set_deterministic_torch(0)

    class MyModel(th.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layers1 = th.nn.Sequential(
                th.nn.Linear(2, 128),
                th.nn.Sigmoid(),
                th.nn.Linear(128, 1)
            )
            self.layers2 = th.nn.Sequential(
                th.nn.Linear(2, 32),
                th.nn.Sigmoid(),
                th.nn.Linear(32, 1)
            )

        def forward(self, x: UdaoInput) -> th.Tensor:

            obj1 = self.layers1(x.features)
            obj2 = self.layers2(x.features)
            output = th.hstack((obj1, obj2))
            assert output.shape[1] == 2
            return output

    objectives = ["obj1", "obj2"]
    loss = WMAPELoss()
    model = MyModel()
    module = UdaoModule(
        model=model,
        objectives=objectives,
        loss=loss,
        metrics=[WeightedMeanAbsolutePercentageError],
        learning_params=LearningParams(init_lr=1e-3, min_lr=1e-5, weight_decay=1e-2)
    )

    tb_logger = TensorBoardLogger("tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-val_WMAPE={val_latency_WeightedMeanAbsolutePercentageError:.2f}",
        auto_insert_metric_name=False,
    )
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
    return model, nn_data_processor

if __name__ == "__main__":

    # data processing and model training
    nn_model, data_processor = set_nn()

    # for optimization
    class Obj1NNModel(th.nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            y = nn_model(x)[:, 0].reshape(-1, 1)
            return y

    class Obj2NNModel(th.nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            y = nn_model(x)[:, 1]
            return th.reshape(y, (-1, 1))

    class Const1(th.nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            norm_x_1 = x.features[:, 0]
            norm_x_2 = x.features[:, 1]
            x_1 = norm_x_1 * 5
            x_2 = norm_x_2 * 3
            y = (x_1 - 5) ** 2 + x_2 ** 2

            return th.reshape(y, (-1, 1))


    class Const2(th.nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            norm_x_1 = x.features[:, 0]
            norm_x_2 = x.features[:, 1]
            x_1 = norm_x_1 * 5
            x_2 = norm_x_2 * 3
            y = (x_1 - 8) ** 2 + (x_2 + 3) ** 2
            return th.reshape(y, (-1, 1))


    objectives = [
        Objective("obj1", minimize=True, function=Obj1NNModel()),
        Objective("obj2", minimize=True, function=Obj2NNModel()),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=Const1(), upper=25),
                   Constraint(function=Const2(), lower=7.7)]

    problem = MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        data_processor=data_processor,
        input_parameters=None,
    )

    so_mogd = MOGD(
        MOGD.Params(
            learning_rate=0.1,
            max_iters=100,
            patience=20,
            multistart=1,
            objective_stress=10,
            device=th.device("cpu"),
        )
    )

    so_grid = GridSearch(GridSearch.Params(n_grids_per_var=[100, 100]))

    # WS
    # w1 = np.linspace(0, 1, num=11, endpoint=True)
    # w2 = 1 - w1
    # ws_pairs = np.vstack((w1, w2)).T
    # ws_algo = WeightedSum(
    #     so_solver=so_grid,
    #     ws_pairs=ws_pairs,
    # )
    # ws_objs, ws_vars = ws_algo.solve(problem=problem)
    # logger.info(f"Found PF-AS solutions of NN: {ws_objs}, {ws_vars}")

    # PF-AS
    spf = SequentialProgressiveFrontier(
        params=SequentialProgressiveFrontier.Params(n_probes=11),
        solver=so_mogd,
    )
    spf_objs, spf_vars = spf.solve(
        problem=problem,
        seed=0
    )
    logger.info(f"Found PF-AS solutions of NN: {spf_objs}, {spf_vars}")

    # PF-AP
    ppf = ParallelProgressiveFrontier(
        params=ParallelProgressiveFrontier.Params(
            processes=1,
            n_grids=2,
            max_iters=4,
        ),
        solver=so_mogd,
    )
    ppf_objs, ppf_vars = ppf.solve(
        problem=problem,
        seed=0,
    )
    logger.info(f"Found PF-AP solutions of NN: {ppf_objs}, {ppf_vars}")

