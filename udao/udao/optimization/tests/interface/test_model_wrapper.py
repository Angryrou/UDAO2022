import pandas as pd
import torch as th

from ....data.extractors.tabular_extractor import TabularFeatureExtractor
from ....data.handler.data_processor import DataProcessor
from ....data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from ....model.model import UdaoModel
from ....model.tests.embedders.dummy_embedder import DummyEmbedder
from ....model.tests.regressors.dummy_regressor import DummyRegressor
from ...interface.model_wrapper import gradient_descent


def test_gradient_descent() -> None:
    data_processor = DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "embedding": TabularFeatureExtractor(columns=["embedding_input"]),
            "tabular_features": TabularFeatureExtractor(
                columns=["feature_input_1", "feature_input_2", "feature_input_3"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
    )
    df = pd.DataFrame(
        {
            "id": list(range(10)),
            "embedding_input": list(range(10)),
            "feature_input_1": [i / 10 for i in range(10)],
            "feature_input_2": [i * 2 for i in range(10)],
            "feature_input_3": [i * 3 for i in range(10)],
            "objective_input": [0 for _ in range(10)],
        }
    )
    df.set_index("id", inplace=True)
    iterator = data_processor.make_iterator(df, list(df.index), split="test")
    model = UdaoModel.from_config(
        embedder_cls=DummyEmbedder,
        regressor_cls=DummyRegressor,
        iterator_shape=iterator.get_iterator_shape(),
        embedder_params={"output_size": 1},
        regressor_params={},
    )
    input_non_decision = {
        "embedding_input": 0,
        "objective_input": 0,
        "feature_input_3": 2.0,
    }
    input_variables = {
        "feature_input_1": [0.5, 0.7, 0.5, 0.7],
        "feature_input_2": [1.0, 1.1, 1.0, 1.1],
    }
    diff = gradient_descent(
        data_processor=data_processor,
        input_non_decision=input_non_decision,
        input_variables=input_variables,
        model=model,
        loss_function=lambda x: th.sum(x**2),
    )
    assert th.allclose(
        diff,
        th.tensor(
            [
                [-1.0000, -1.0000, 0.0000],
                [-1.0000, -1.0000, 0.0000],
                [-1.0000, -1.0000, 0.0000],
                [-1.0000, -1.0000, 0.0000],
            ]
        ),
    )
