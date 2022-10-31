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
from torch.optim import SGD

from pyDOE import lhs
from sklearn.utils import shuffle
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
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

    def get_samples(self, n_samples, criterion="maximin", debug=False, random_state=None):
        """
        generate n_samples samples via LHS
        :param n_samples:
        :param criterion:
        :param debug:
        :param random_state:
        :return:
        """
        assert n_samples > 1
        if random_state is None:
            random_state = self.seed
        else:
            assert isinstance(random_state, int)
            np.random.seed(random_state)

        # get the internal samples
        samples = lhs(self.n_knobs, samples=n_samples, criterion=criterion)
        samples = shuffle(samples, random_state=random_state)
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
    def __init__(self, knobs, seed=42, debug=False):
        super(BOSampler, self).__init__(knobs, seed)
        self.debug = debug

    def get_samples(self, n_samples, observed_inputs=None, observed_outputs=None,
                    optimizer="default", lr=1, epochs=100):
        """
        get 1 sample from based on BO based on current observations.
        :param n_samples: number of additional samples needed, usually set to 1 in BO
        :param observed_inputs: a dataframe of observed knobs
        :param observed_outputs: a dataframe of observed values
        :param optimizer: name of the optimizer
        :param lr: used when optimizer=SGD
        :param epochs: when optimizer=SGD
        :return: n_samples configuration recommended by BO
        """
        assert isinstance(observed_inputs, np.ndarray) and isinstance(observed_outputs, np.ndarray)
        if observed_outputs.ndim == 1:
            observed_outputs = observed_outputs.reshape(-1, 1)
        else:
            assert observed_outputs.ndim == 2
        assert observed_inputs.shape[1] == self.n_knobs and len(observed_inputs) == len(observed_outputs)
        assert observed_inputs.min() >= 0 and observed_inputs.max() <= 1, \
            "the observed inputs we take should be normalized to 0-1"

        X_tr = th.from_numpy(observed_inputs).to(th.float)
        y_tr = th.from_numpy(
            (observed_outputs - observed_outputs.min()) / (observed_outputs.max() - observed_outputs.min())
        ).to(th.float)
        gp = SingleTaskGP(X_tr, - y_tr)
        if optimizer == "default":
            # L-BFGS-B
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(X_tr)
            fit_gpytorch_model(mll)
        elif optimizer == "SGD":
            # NUM_EPOCHS, LR are heuristically chosen by the local testing of our dataset
            th.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll = mll.to(X_tr)
            optimizer = SGD([{'params': gp.parameters()}], lr=lr)
            gp.train()
            # TODO: the current method works fine for 4k data points, could be extended to a mini-batch training
            #       when the number of data points is much larger, e.g. 400k
            for epoch in range(epochs):

                # clear gradients
                optimizer.zero_grad()
                # forward pass through the model to obtain the output MultivariateNormal
                output = gp(X_tr)
                # Compute negative marginal log likelihood
                loss = - mll(output, gp.train_targets)
                # back prop gradients
                loss.backward()
                # print every 10 iterations
                if self.debug and (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch + 1:>3}/{epochs} - Loss: {loss.item():>4.3f} "
                        f"noise: {gp.likelihood.noise.item():>4.3f}"
                    )
                optimizer.step()

        else:
            raise NotImplementedError(optimizer)

        bounds = th.stack([th.zeros(self.n_knobs), th.ones(self.n_knobs)])

        try:
            reco_samples, _ = optimize_acqf(
                # UCB: https://people.eecs.berkeley.edu/~kandasamy/talks/electrochem-bo-slides.pdf
                # UpperConfidenceBound(gp, beta=UCB_BETA),
                # qEI
                qExpectedImprovement(gp, best_f=0.2),
                bounds=bounds,
                q=n_samples,
                num_restarts=5,
                raw_samples=100
            )
        except Exception as e:
            raise Exception(f"got error when recommending: {e}")

        return reco_samples.numpy()
