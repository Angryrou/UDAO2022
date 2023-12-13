from typing import Dict, List

import numpy as np
import pytest
import torch as th
import torch.nn as nn

from ....data.containers.tabular_container import TabularContainer
from ....data.extractors.tabular_extractor import TabularFeatureExtractor
from ....data.handler.data_processor import DataProcessor
from ....data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from ....data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from ....model.utils.utils import set_deterministic_torch
from ....utils.interfaces import UdaoInput
from ...concepts import EnumVariable, FloatVariable, IntegerVariable, Variable
from ...concepts.constraint import Constraint
from ...concepts.objective import Objective
from ...concepts.utils import ModelComponent
from ...solver.mogd import MOGD


class SimpleModel1(nn.Module):
    def forward(self, x: UdaoInput, wl_id: None = None) -> th.Tensor:
        return x.feature_input[:, :1]


class SimpleModel2(nn.Module):
    def forward(self, x: UdaoInput, wl_id: None = None) -> th.Tensor:
        return x.feature_input[:, 1:]


@pytest.fixture()
def data_processor() -> DataProcessor:
    class TabularFeaturePreprocessor(StaticFeaturePreprocessor):
        def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
            tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 1
            tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] - 2
            return tabular_feature

        def inverse_transform(
            self, tabular_feature: TabularContainer
        ) -> TabularContainer:
            tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 1
            tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] + 2
            return tabular_feature

    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "embedding_features": TabularFeatureExtractor(columns=["embedding_input"]),
            "tabular_features": TabularFeatureExtractor(
                columns=["v1", "v2"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
        feature_preprocessors={"tabular_features": [TabularFeaturePreprocessor()]},
    )


@pytest.fixture()
def mogd() -> MOGD:
    set_deterministic_torch(42)

    params = MOGD.Params(
        learning_rate=0.1,
        weight_decay=0.1,
        max_iters=100,
        patient=10,
        seed=0,
        multistart=10,
        stress=0.1,
    )
    mogd = MOGD(params)

    return mogd


@pytest.fixture()
def data_processor_paper() -> DataProcessor:
    return DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "embedding_features": TabularFeatureExtractor(columns=["embedding_input"]),
            "tabular_features": TabularFeatureExtractor(
                columns=["v1"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
    )


class PaperModel1(nn.Module):
    def forward(self, x: UdaoInput, wl_id: None = None) -> th.Tensor:
        return th.reshape(2400 / (x.feature_input[:, 0]), (-1, 1))


class PaperModel2(nn.Module):
    def forward(self, x: UdaoInput, wl_id: None = None) -> th.Tensor:
        return th.reshape(x.feature_input[:, 0], (-1, 1))


@pytest.fixture()
def paper_mogd() -> MOGD:
    set_deterministic_torch(42)

    params = MOGD.Params(
        learning_rate=1,
        weight_decay=0.0,
        max_iters=100,
        patient=10,
        seed=0,
        multistart=10,
        stress=10,
    )
    mogd = MOGD(params)

    return mogd


class TestMOGD:
    @pytest.mark.parametrize(
        "gpu, expected_obj, expected_vars",
        [
            (False, 1, {"v1": 1.0, "v2": 2.0}),
            (True, 0.728246, [0.07, 2.15]),
        ],
    )
    def test_solve(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        gpu: bool,
        expected_obj: float,
        expected_vars: Dict[str, float],
    ) -> None:
        if gpu and not th.cuda.is_available():
            pytest.skip("Skip GPU test")
        mogd.device = th.device("cuda") if gpu else th.device("cpu")
        objective_function = ModelComponent(
            data_processor=data_processor,
            model=SimpleModel1(),  # type: ignore
        )
        optimal_obj, optimal_vars = mogd.solve(
            objective=Objective(
                "obj1",
                direction_type="MAX",
                function=objective_function,
                lower=0,
                upper=2,
            ),
            variables={"v1": FloatVariable(0, 1), "v2": IntegerVariable(2, 3)},
            constraints=[
                Constraint(
                    lower=0,
                    upper=1,
                    function=ModelComponent(
                        data_processor=data_processor, model=SimpleModel2()
                    ),
                    stress=10,
                )
            ],
            input_parameters={"embedding_input": 0, "objective_input": 0},
        )
        assert optimal_obj is not None
        np.testing.assert_array_almost_equal(
            optimal_obj, np.array(expected_obj), decimal=5
        )
        assert optimal_vars == expected_vars

    @pytest.mark.parametrize(
        "variable, expected_variable",
        [(FloatVariable(0, 24), {"v1": 16}), (IntegerVariable(0, 24), {"v1": 16})],
    )
    def test_solve_paper(
        self,
        paper_mogd: MOGD,
        variable: Variable,
        expected_variable: Dict[str, float],
        data_processor_paper: DataProcessor,
    ) -> None:
        optimal_obj, optimal_vars = paper_mogd.solve(
            objective=Objective(
                "obj1",
                "MIN",
                function=ModelComponent(
                    model=PaperModel1(), data_processor=data_processor_paper
                ),
                lower=100,
                upper=200,
            ),
            variables={"v1": variable},
            constraints=[
                Constraint(
                    lower=8,
                    upper=16,
                    function=ModelComponent(
                        model=PaperModel2(), data_processor=data_processor_paper
                    ),
                    stress=10,
                )
            ],
            input_parameters={"embedding_input": 0, "objective_input": 0},
        )

        assert optimal_obj is not None
        np.testing.assert_allclose([optimal_obj], [150], rtol=1e-3)
        assert optimal_vars is not None
        assert len(optimal_vars) == 1
        np.testing.assert_allclose(
            [optimal_vars["v1"]], [expected_variable["v1"]], rtol=1e-3
        )

    def test_solve_no_constraints(
        self, mogd: MOGD, data_processor: DataProcessor
    ) -> None:
        objective_function = ModelComponent(
            model=lambda x: th.reshape(  # type: ignore
                x.feature_input[:, 0] ** 2 + x.feature_input[:, 1] ** 2, (-1, 1)
            ),
            data_processor=data_processor,
        )
        optimal_obj, optimal_vars = mogd.solve(
            objective=Objective(
                name="obj1",
                direction_type="MAX",
                function=objective_function,
            ),
            variables={"v1": FloatVariable(0, 1), "v2": IntegerVariable(2, 3)},
            input_parameters={"embedding_input": 0, "objective_input": 0},
        )

        assert optimal_obj is not None
        np.testing.assert_array_equal(optimal_obj, 2)
        assert optimal_vars is not None
        assert optimal_vars == {"v1": 1, "v2": 3}

    @pytest.mark.parametrize(
        "objective_values, expected_loss",
        [
            # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
            (th.tensor([0.5]), th.tensor([-0.25])),
            # (-0.2 / 2 - 0.5)**2 + stress (0.1) = 0.46
            (th.tensor([-0.2]), th.tensor([0.46])),
            (th.tensor([[0.5], [0.3], [-0.2]]), th.tensor([[-0.25], [-0.15], [0.46]])),
        ],
    )
    def test__objective_loss_with_bounds(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        objective_values: th.Tensor,
        expected_loss: th.Tensor,
    ) -> None:
        objective = Objective(
            "obj1",
            "MAX",
            function=ModelComponent(
                data_processor=data_processor, model=SimpleModel1()
            ),
            lower=0,
            upper=2,
        )
        loss = mogd.objective_loss(objective_values, objective)
        # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
        assert th.equal(loss.cpu(), expected_loss)

    @pytest.mark.parametrize(
        "objective_values, expected_loss",
        [
            # direction * 0.5**2
            (th.tensor([0.5]), th.tensor([-0.25])),
            # direction * (-O.2)**2
            (th.tensor([-0.2]), th.tensor([-0.04])),
            (th.tensor([[0.5], [0.3], [-0.2]]), th.tensor([[-0.25], [-0.09], [-0.04]])),
        ],
    )
    def test__objective_loss_without_bounds(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        objective_values: th.Tensor,
        expected_loss: th.Tensor,
    ) -> None:
        objective = Objective(
            "obj1",
            "MAX",
            function=ModelComponent(
                data_processor=data_processor, model=SimpleModel1()
            ),
        )
        loss = mogd.objective_loss(objective_values, objective)
        # 0.5 /2 (normalized) * direction (-1 for max) = -0.25
        assert th.allclose(loss, expected_loss)

    @pytest.mark.parametrize(
        "constraint_values, expected_loss",
        [
            (
                [th.tensor([1.1]), th.tensor([1.1]), th.tensor([3.5])],
                # 0.6**2 + 10+ 0 + 0.5**2 + 1000
                th.tensor([1010.61]),
            ),
        ],
    )
    def test_constraints_loss(
        self,
        mogd: MOGD,
        data_processor: DataProcessor,
        constraint_values: List[th.Tensor],
        expected_loss: th.Tensor,
    ) -> None:
        constraints = [
            Constraint(
                lower=0,
                upper=1,
                function=ModelComponent(
                    data_processor=data_processor, model=SimpleModel1()
                ),
                stress=10,
            ),
            Constraint(
                lower=0,
                upper=2,
                function=ModelComponent(
                    data_processor=data_processor, model=SimpleModel2()
                ),
                stress=100,
            ),
            Constraint(
                upper=3,
                function=ModelComponent(
                    data_processor=data_processor, model=SimpleModel2()
                ),
                stress=1000,
            ),
        ]
        loss = mogd.constraints_loss(constraint_values, constraints)
        assert th.allclose(loss, expected_loss)

    def test_get_meshed_categorical_variables(self, mogd: MOGD) -> None:
        variables = {
            "v1": IntegerVariable(2, 3),
            "v2": EnumVariable([4, 5]),
            "v3": EnumVariable([10, 20]),
        }
        meshed_variables = mogd.get_meshed_categorical_vars(variables=variables)
        assert meshed_variables is not None
        np.testing.assert_array_equal(
            meshed_variables, [[4, 10], [5, 10], [4, 20], [5, 20]]
        )
