from typing import Dict, Sequence

import numpy as np
import pytest
import torch as th
from torch import nn

from udao.data.containers.tabular_container import TabularContainer
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_processor import DataProcessor
from udao.data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from udao.data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from udao.utils.interfaces import UdaoInput
from udao.optimization.concepts import Constraint, FloatVariable, IntegerVariable, Objective, Variable
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.concepts.utils import ModelComponent
from udao.optimization.concepts.utils import InputParameters, InputVariables
from udao.optimization.soo.mogd import MOGD

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

@pytest.fixture
def hcf_problem() -> MOProblem:
    def obj_func1(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return th.tensor(
            4 * input_variables["v1"] ** 2 + 4 * input_variables["v2"] ** 2
            + (input_parameters or {}).get("count", 0)
        )

    def obj_func2(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return th.tensor(
            (input_variables["v1"] - 5) ** 2 + (input_variables["v2"] - 5) ** 2
            + (input_parameters or {}).get("count", 0)
        )

    objectives = [
        Objective(
            "obj1",
            function=obj_func1,
            direction_type="MIN",
        ),
        Objective("obj2", function=obj_func2, direction_type="MIN"),
    ]

    def constraint_func1(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return th.tensor(
            (input_variables["v1"] - 5) ** 2 + input_variables["v2"] ** 2
            - (input_parameters or {}).get("count", 0)
        )

    def constraint_func2(
        input_variables: InputVariables, input_parameters: InputParameters = None
    ) -> th.Tensor:
        return th.tensor(
            (input_variables["v1"] - 8) ** 2 + (input_variables["v2"] + 3) ** 2
            - (input_parameters or {}).get("count", 0)
        )

    constraints = [Constraint(function=constraint_func1, upper=25),
                   Constraint(function=constraint_func2, lower=7.7)]
    return MOProblem(
        objectives=objectives,
        constraints=constraints,
        variables={"v1": FloatVariable(0, 5), "v2": FloatVariable(0, 3)},
        input_parameters={"count": 1},
    )

class TabularFeaturePreprocessor(StaticFeaturePreprocessor):
    def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 5
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] / 3
        return tabular_feature

    def inverse_transform(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 5
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] * 3
        return tabular_feature

@pytest.fixture()
def data_processor() -> DataProcessor:
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

class Obj1(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        norm_x_1 = x.feature_input[:, 0]
        norm_x_2 = x.feature_input[:, 1]
        x_1 = norm_x_1 * 5
        x_2 = norm_x_2 * 3
        y = 4 * x_1 ** 2 + 4 * x_2 ** 2
        return th.reshape(y, (-1, 1))

class Obj2(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        norm_x_1 = x.feature_input[:, 0]
        norm_x_2 = x.feature_input[:, 1]
        x_1 = norm_x_1 * 5
        x_2 = norm_x_2 * 3
        y = (x_1 - 5) ** 2 + (x_2 - 5) ** 2
        return th.reshape(y, (-1, 1))

class Const1(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        norm_x_1 = x.feature_input[:, 0]
        norm_x_2 = x.feature_input[:, 1]
        x_1 = norm_x_1 * 5
        x_2 = norm_x_2 * 3
        y = (x_1 - 5) ** 2 + x_2 ** 2
        return th.reshape(y, (-1, 1))

class Const2(nn.Module):
    def forward(self, x: UdaoInput) -> th.Tensor:
        norm_x_1 = x.feature_input[:, 0]
        norm_x_2 = x.feature_input[:, 1]
        x_1 = norm_x_1 * 5
        x_2 = norm_x_2 * 3
        y = (x_1 - 8) ** 2 + (x_2 + 3) ** 2
        return th.reshape(y, (-1, 1))

@pytest.fixture
def simple_nn_problem(data_processor: DataProcessor) -> MOProblem:

    # class ObjModel1(nn.Module):
    #     def forward(self, x: UdaoInput) -> th.Tensor:
    #         obj1 = Obj1()
    #         y = obj1.forward(x)
    #         return th.reshape(y, (-1, 1))
    #
    # class ObjModel2(nn.Module):
    #     def forward(self, x: UdaoInput) -> th.Tensor:
    #         obj2 = Obj2()
    #         y = obj2.forward(x)
    #         return th.reshape(y, (-1, 1))
    #
    # class ConstModel1(nn.Module):
    #     def forward(self, x: UdaoInput) -> th.Tensor:
    #         const1 = Const1()
    #         y = const1.forward(x)
    #         return th.reshape(y, (-1, 1))
    #
    # class ConstModel2(nn.Module):
    #     def forward(self, x: UdaoInput) -> th.Tensor:
    #         obj1 = Obj1()
    #         y = obj1.forward(x)
    #         return th.reshape(y, (-1, 1))

    objectives = [
        Objective("obj1", "MIN", ModelComponent(data_processor, Obj1())),
        Objective("obj2", "MIN", ModelComponent(data_processor, Obj2())),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=ModelComponent(data_processor, Const1()), upper=25),
                  Constraint(function=ModelComponent(data_processor, Const2()), lower=7.7)]
    input_parameters = {
        "embedding_input": 1,
        "objective_input": 1,
    }
    return MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        input_parameters=input_parameters,
    )

@pytest.fixture
def nn_problem(data_processor: DataProcessor) -> MOProblem:
    n_input = 2
    n_out = 1
    n_hid = 128
    n_epoch = 20
    learning_rate = 0.01

    def fit(X_train):
        model = nn.Sequential(nn.Linear(n_input, n_hid),
                              nn.Sigmoid(),
                              nn.Linear(n_hid, n_out),
                              # nn.Sigmoid()
                              )
        loss_function = nn.MSELoss()
        optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
        losses = []
        model_dict = {}
        for obj in ["obj1", "obj2", "const1", "const2"]:
            for epoch in range(n_epoch):
                # get forward prediction
                pred_y = model(X_train)
                loss = loss_function(pred_y, y_dict[obj])
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                # for name, param in model.named_parameters():
                #     print(name, param.grad)
                optimizer.step()
            model_dict[obj] = model
        return model_dict

    X_train, y_dict = get_raw_data()
    model_dict = fit(X_train)

    class ObjModel1(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = 4 * x_1 ** 2 + 4 * x_2 ** 2
            y = model_dict["obj1"](x.feature_input)
            return th.reshape(y, (-1, 1))

    class ObjModel2(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 5) ** 2 + (x_2 - 5) ** 2
            y = model_dict["obj2"](x.feature_input)
            return th.reshape(y, (-1, 1))

    class ConstModel1(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 5) ** 2 + x_2 ** 2
            y = model_dict["const1"](x.feature_input)
            return th.reshape(y, (-1, 1))

    class ConstModel2(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 8) ** 2 + (x_2 + 3) ** 2
            y = model_dict["const2"](x.feature_input)
            return th.reshape(y, (-1, 1))

    objectives = [
        Objective("obj1", "MIN", ModelComponent(data_processor, ObjModel1())),
        Objective("obj2", "MIN", ModelComponent(data_processor, ObjModel2())),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=ModelComponent(data_processor, ConstModel1()), upper=25),
                  Constraint(function=ModelComponent(data_processor, ConstModel2()), lower=7.7)]
    input_parameters = {
        "embedding_input": 1,
        "objective_input": 1,
    }
    return MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        input_parameters=input_parameters,
    )

@pytest.fixture
def gp_problem(data_processor: DataProcessor) -> MOProblem:
    length_scale = 1
    magnitude = 1

    def fit(X_train, y_dict):
        '''
        fit GPR models
        :param X_train: ndarray(n_training_samples, n_vars), input to train GPR models, should be normalized
        :return:
                X_train: Tensor(n_training_samples, n_vars), input to train GPR models
                y_tensor_dict: dict, key is objective names, values are training java_data of true objective values with Tensor(n_training_samples,) format
                K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        '''
        ridge = 1
        if X_train.ndim != 2:
            raise Exception("x_train should have 2 dimensions! X_dim:{}"
                            .format(X_train.ndim))

        sample_size = X_train.shape[0]
        if np.isscalar(ridge):
            ridge = th.ones(sample_size) * ridge
        else:
            raise Exception(f"rideg is not set properly!")

        # assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1

        K = magnitude * th.exp(-gs_dist(X_train, X_train) / length_scale) + th.diag(ridge)
        K_inv = K.inverse()

        y_tensor_dict = {}
        for obj in ["obj1", "obj2", "const1", "const2"]:
            y_tensor_dict[obj] = y_dict[obj]
        return [X_train, y_tensor_dict, K_inv]

    def gs_dist(x1, x2):
        '''
        calculate distances between each training sample
        :param x1: Tensor(n_training_samples, n_vars), training input java_data
        :param x2: Tensor(n_training_samples or n_x, n_vars), training input java_data or test input java_data
        :return:
                dist: Tensor(n_training_samples, n_training_samples), each element shows distance between each training sample
        '''
        # e.g.
        # x1.shape = (m, 12)
        # x2.shape = (n, 12)
        # K(x1, x2).shape = (m, n)
        assert x1.shape[1] == x2.shape[1]
        comm_dim = x1.shape[1]
        dist = th.norm(x1.reshape(-1, 1, comm_dim) - x2.reshape(1, -1, comm_dim), dim=2)
        return dist

    def objective(X_test, X_train, y_train, K_inv):
        '''
        call GPR model to get objective values
        :param X_test: Tensor(n_x, n_vars), input of the predictive model, where n_x shows the number of input variables
        :param X_train: Tensor(n_training_samples, n_vars), input to train GPR models
        :param y_train: Tensor(n_training_samples,), training java_data of true objective values
        :param K_inv: Tensor(n_training_samples, n_training_samples), inversion of the covariance matrix
        :return:
                yhat: Tensor(n_x,), the prediction of the objective
        '''
        if X_test.ndimension() == 1:
            X_test = X_test.reshape(1, -1)
        K2 = magnitude * th.exp(-gs_dist(X_train, X_test) / length_scale)
        K2_trans = K2.t()
        yhat = th.matmul(K2_trans, th.matmul(K_inv, y_train))
        return yhat

    X_train, y_dict = get_raw_data()
    gp_model = fit(X_train, y_dict)

    class ObjModel1(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = 4 * x_1 ** 2 + 4 * x_2 ** 2
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["obj1"]
            y = objective(x.feature_input, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))

    class ObjModel2(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 5) ** 2 + (x_2 - 5) ** 2
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["obj2"]
            y = objective(x.feature_input, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))

    class ConstModel1(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 5) ** 2 + x_2 ** 2
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["const1"]
            y = objective(x.feature_input, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))

    class ConstModel2(nn.Module):
        def forward(self, x: UdaoInput) -> th.Tensor:
            # norm_x_1 = x.feature_input[:, 0]
            # norm_x_2 = x.feature_input[:, 1]
            # x_1 = norm_x_1 * 5
            # x_2 = norm_x_2 * 3
            # y = (x_1 - 8) ** 2 + (x_2 + 3) ** 2
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["const2"]
            y = objective(x.feature_input, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))

    objectives = [
        Objective("obj1", "MIN", ModelComponent(data_processor, ObjModel1())),
        Objective("obj2", "MIN", ModelComponent(data_processor, ObjModel2())),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=ModelComponent(data_processor, ConstModel1()), upper=25),
                  Constraint(function=ModelComponent(data_processor, ConstModel2()), lower=7.7)]
    input_parameters = {
        "embedding_input": 1,
        "objective_input": 1,
    }
    return MOProblem(
        objectives=objectives,
        variables=variables,
        constraints=constraints,
        input_parameters=input_parameters,
    )

@pytest.fixture
def mogd() -> MOGD:
    return MOGD(
        MOGD.Params(
            learning_rate=0.1,
            weight_decay=0,
            max_iters=100,
            patience=10,
            multistart=10,
            objective_stress=10,
            seed=0,
            device=th.device("cpu"),
        )
    )

def get_raw_data():
    rand_x_1 = th.from_numpy(np.random.uniform(0, 5, 1000))
    rand_x_2 = th.from_numpy(np.random.uniform(0, 3, 1000))
    X_train = th.vstack((rand_x_1, rand_x_2)).T

    obj1 = 4 * rand_x_1 ** 2 + 4 * rand_x_2 ** 2
    obj2 = (rand_x_1 - 5) ** 2 + (rand_x_2 - 5) ** 2
    const1 = (rand_x_1 - 5) ** 2 + rand_x_2 ** 2
    const2 = (rand_x_1 - 8) ** 2 + (rand_x_2 + 3) ** 2

    avail_inds1 = th.where(const1 <= 25)
    avail_inds2 = th.where(const2 >= 7.7)
    avail_inds = np.intersect1d(avail_inds1, avail_inds2)
    name_list = ["obj1", "obj2", "const1", "const2"]
    value_list = [obj1[avail_inds], obj2[avail_inds], const1[avail_inds], const2[avail_inds]]
    y_dict = {i: v.to(th.float32) for i, v in zip(name_list, value_list)}

    norm_X_train = X_train[avail_inds]
    norm_X_train[:, 0] = norm_X_train[:, 0] / 5
    norm_X_train[:, 1] = norm_X_train[:, 1] / 3
    return norm_X_train.to(th.float32), y_dict



