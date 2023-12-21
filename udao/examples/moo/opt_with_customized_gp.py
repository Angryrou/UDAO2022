from typing import Dict

import numpy as np
import torch as th

from udao.data.containers.tabular_container import TabularContainer

from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_processor import DataProcessor
from udao.data.preprocessors.base_preprocessor import StaticFeaturePreprocessor
from udao.data.tests.iterators.dummy_udao_iterator import DummyUdaoIterator
from udao.optimization.concepts import Constraint, FloatVariable, Objective, Variable
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.moo.weighted_sum import WeightedSum
from udao.optimization.moo.progressive_frontier import SequentialProgressiveFrontier
from udao.optimization.moo.progressive_frontier import ParallelProgressiveFrontier
from udao.optimization.soo.grid_search_solver import GridSearch
from udao.optimization.soo.mogd import MOGD
from udao.utils.interfaces import UdaoEmbedInput
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

def set_gp():
    length_scale = 1
    magnitude = 1

    def get_raw_data():
        n = 1000
        np.random.seed(0)
        rand_x_1 = th.from_numpy(np.random.uniform(0, 5, n))
        rand_x_2 = th.from_numpy(np.random.uniform(0, 3, n))
        raw_X_train = th.vstack((rand_x_1, rand_x_2)).T

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

        norm_X_train = raw_X_train[avail_inds]
        norm_X_train[:, 0] = norm_X_train[:, 0] / 5
        norm_X_train[:, 1] = norm_X_train[:, 1] / 3
        return norm_X_train.to(th.float32), y_dict

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

    return X_train, y_dict, gp_model, objective

class TabularFeaturePreprocessor(StaticFeaturePreprocessor):
    def preprocess(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] / 5
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] / 3
        return tabular_feature

    def inverse_transform(self, tabular_feature: TabularContainer) -> TabularContainer:
        tabular_feature.data.loc[:, "v1"] = tabular_feature.data["v1"] * 5
        tabular_feature.data.loc[:, "v2"] = tabular_feature.data["v2"] * 3
        return tabular_feature

if __name__ == "__main__":

    # data processing
    data_processor = DataProcessor(
        iterator_cls=DummyUdaoIterator,
        feature_extractors={
            "tabular_features": TabularFeatureExtractor(
                columns=["v1", "v2"],
            ),
            "objectives": TabularFeatureExtractor(columns=["objective_input"]),
        },
        feature_preprocessors={"tabular_features": [TabularFeaturePreprocessor()]},
    )

    # for customzied GP model
    X_train, y_dict, gp_model, objective = set_gp()

    # for optimization
    class ObjGPModel1(th.nn.Module):
        def forward(self, x: UdaoEmbedInput) -> th.Tensor:
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["obj1"]
            y = objective(x.features, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))


    class ObjGPModel2(th.nn.Module):
        def forward(self, x: UdaoEmbedInput) -> th.Tensor:
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["obj2"]
            y = objective(x.features, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))


    class ConstGPModel1(th.nn.Module):
        def forward(self, x: UdaoEmbedInput) -> th.Tensor:
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["const1"]
            y = objective(x.features, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))


    class ConstGPModel2(th.nn.Module):
        def forward(self, x: UdaoEmbedInput) -> th.Tensor:
            X_train, y_dict, K_inv = gp_model
            y_train = y_dict["const2"]
            y = objective(x.features, X_train, y_train, K_inv).view(-1, 1)
            return th.reshape(y, (-1, 1))


    objectives = [
        Objective("obj1", minimize=True, function=ObjGPModel1()),
        Objective("obj2", minimize=True, function=ObjGPModel2()),
    ]
    variables: Dict[str, Variable] = {
        "v1": FloatVariable(0, 5),
        "v2": FloatVariable(0, 3),
    }
    constraints = [Constraint(function=ConstGPModel1(), upper=25),
                   Constraint(function=ConstGPModel2(), lower=7.7)]

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
    w1 = np.linspace(0, 1, num=11, endpoint=True)
    w2 = 1 - w1
    ws_pairs = np.vstack((w1, w2)).T
    ws_algo = WeightedSum(
        so_solver=so_grid,
        ws_pairs=ws_pairs,
    )
    ws_objs, ws_vars = ws_algo.solve(problem=problem)
    logger.info(f"Found PF-AS solutions of customized GP: {ws_objs}, {ws_vars}")

    # PF-AS
    spf = SequentialProgressiveFrontier(
        params=SequentialProgressiveFrontier.Params(n_probes=11),
        solver=so_mogd,
    )
    spf_objs, spf_vars = spf.solve(
        problem=problem,
        seed=0
    )
    logger.info(f"Found PF-AS solutions of customized GP: {spf_objs}, {spf_vars}")

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
    logger.info(f"Found PF-AP solutions of customized GP: {ppf_objs}, {ppf_vars}")
