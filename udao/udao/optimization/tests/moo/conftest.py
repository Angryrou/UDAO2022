from typing import Dict, Sequence

import pytest
import torch as th
from torch import nn

from ....data.containers.tabular_container import TabularContainer
from ....data.extractors.tabular_extractor import TabularFeatureExtractor
from ....data.handler.data_processor import DataProcessor
from ....data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from ....data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from ....utils.interfaces import UdaoEmbedInput
from ...concepts import Constraint, FloatVariable, IntegerVariable, Objective, Variable
from ...concepts.problem import MOProblem
from ...soo.mogd import MOGD


class ObjModel1(nn.Module):
    def forward(self, x: UdaoEmbedInput) -> th.Tensor:
        return th.reshape(x.features[:, 0] ** 2, (-1, 1))


class ObjModel2(nn.Module):
    def forward(self, x: UdaoEmbedInput) -> th.Tensor:
        return th.reshape(x.features[:, 1] ** 2, (-1, 1))


class ComplexObj1(nn.Module):
    def forward(self, x: UdaoEmbedInput) -> th.Tensor:
        return th.reshape(x.features[:, 0] ** 2 - x.features[:, 1] ** 2, (-1, 1))


class ComplexObj2(nn.Module):
    def forward(self, x: UdaoEmbedInput) -> th.Tensor:
        return th.reshape(x.features[:, 0] ** 2 + x.features[:, 1] ** 2, (-1, 1))


class TabularFeaturePreprocessor(StaticFeaturePreprocessor):
    def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 1
        tabular_feature.data.loc[:, "v2"] = (tabular_feature.data["v2"] - 1) / 6
        return tabular_feature

    def inverse_transform(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 1
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] * 6 + 1
        return tabular_feature


@pytest.fixture()
def data_processor() -> DataProcessor:
    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "tabular_features": TabularFeatureExtractor(
                columns=["v1", "v2"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
        feature_preprocessors={"tabular_features": [TabularFeaturePreprocessor()]},
    )


@pytest.fixture
def mogd() -> MOGD:
    return MOGD(
        MOGD.Params(
            learning_rate=0.1,
            max_iters=100,
            patience=10,
            multistart=10,
            objective_stress=10,
            constraint_stress=1e5,
            device=th.device("cpu"),
        )
    )


@pytest.fixture
def two_obj_problem(data_processor: DataProcessor) -> MOProblem:
    objectives = [
        Objective("obj1", minimize=True, function=ObjModel1()),
        Objective("obj2", minimize=True, function=ObjModel2()),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 1),
        "v2": IntegerVariable(1, 7),
    }
    constraints: Sequence[Constraint] = []

    return MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        data_processor=data_processor,
        input_parameters=None,
    )


@pytest.fixture
def three_obj_problem(
    two_obj_problem: MOProblem, data_processor: DataProcessor
) -> MOProblem:
    return MOProblem(
        objectives=[
            Objective("obj1", minimize=False, function=ObjModel1()),
            Objective("obj2", minimize=False, function=ObjModel2()),
            Objective("obj3", minimize=False, function=ComplexObj2()),
        ],
        variables=two_obj_problem.variables,
        constraints=two_obj_problem.constraints,
        input_parameters=two_obj_problem.input_parameters,
        data_processor=data_processor,
    )
