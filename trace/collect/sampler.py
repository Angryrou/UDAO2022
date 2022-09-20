# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: Latin Hyper Sampling method (LHS) and Bayesian Optimization (BO) method to generate configurations
#
# Created at 9/16/22

from abc import ABCMeta, abstractmethod
from utils.parameters import UCB_BETA
from utils.data.configurations import KnobUtils
import random
import numpy as np
import torch as th

from pyDOE import lhs
from sklearn.utils import shuffle
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


class BaseSampler(metaclass=ABCMeta):
    def __init__(self, knobs, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.knobs = knobs
        self.knobs_min = np.array([k.min for k in knobs])
        self.knobs_max = np.array([k.max for k in knobs])
        self.n_knobs = len(knobs)
        self.seed = seed

    @abstractmethod
    def get_samples(self, n_samples, **kwargs):
        pass


class LHSSampler(BaseSampler):
    def __init__(self, knobs, seed=42):
        super(LHSSampler, self).__init__(knobs, seed)

    def get_samples(self, n_samples, criterion="maximin", debug=False):
        """
        generate n_samples samples via LHS
        :param n_samples:
        :param criterion:
        :param debug:
        :return:
        """
        assert n_samples > 1

        # get the internal samples
        samples = lhs(self.n_knobs, samples=n_samples, criterion=criterion)
        samples = shuffle(samples, random_state=self.seed)
        # get decoded configurations in a DataFrame
        knob_df = KnobUtils.knob_denormalize(samples, self.knobs)
        # drop duplicated configurations after rounding in the decoding
        knob_df = knob_df.drop_duplicates()
        knob_df.index = knob_df.apply(lambda x: KnobUtils.knobs2sign(x, self.knobs), axis=1)
        if debug:
            return samples, knob_df
        else:
            return knob_df


class BOSampler(BaseSampler):
    def __init__(self, knobs, seed=42):
        super(BOSampler, self).__init__(knobs, seed)

    def get_samples(self, n_samples, observed_inputs=None, observed_outputs=None):
        """
        get 1 sample from based on BO based on current observations.
        :param n_samples: number of additional samples needed, usually set to 1 in BO
        :param observed_inputs: a dataframe of observed knobs
        :param observed_outputs: a dataframe of observed values
        :return: n_samples configuration recommended by BO
        """
        assert n_samples == 1
        assert isinstance(observed_inputs, np.ndarray) and isinstance(observed_outputs, np.ndarray)
        if observed_outputs.ndim == 1:
            observed_outputs = observed_outputs.reshape(-1, 1)
        else:
            assert observed_outputs.ndim == 2
        assert observed_inputs.shape[1] == self.n_knobs and len(observed_inputs) == len(observed_outputs)
        assert observed_inputs.min() >= 0 and observed_inputs.max() <= 1, \
            "the observed inputs we take should be normalized to 0-1"

        X_tr = th.from_numpy(observed_inputs)
        y_tr = th.from_numpy(observed_outputs - observed_outputs.min() /
                             (observed_outputs.max() - observed_outputs.min()))
        gp = SingleTaskGP(X_tr, - y_tr)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        bounds = th.stack([th.zeros(self.n_knobs), th.ones(self.n_knobs)])

        try:
            reco_sample, _ = optimize_acqf(
                # UCB: https://people.eecs.berkeley.edu/~kandasamy/talks/electrochem-bo-slides.pdf
                UpperConfidenceBound(gp, beta=UCB_BETA),
                bounds=bounds,
                q=n_samples,
                num_restarts=5,
                raw_samples=20
            )
        except Exception as e:
            raise Exception(f"got error when recommending: {e}")

        return reco_sample