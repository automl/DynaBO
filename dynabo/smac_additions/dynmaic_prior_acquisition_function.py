from typing import Dict, List

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import FloatHyperparameter
from smac.acquisition.function import (
    LCB,
    TS,
    AbstractAcquisitionFunction,
    IntegratedAcquisitionFunction,
    PriorAcquisitionFunction,
)
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigSpacePdfWrapper:
    def __init__(self, configspace: ConfigurationSpace, decay_beta: float, prior_start: int, prior_floor: float, discretize: bool, discrete_bins_factor: float, dynabo: bool):
        self.configspace = configspace
        self.decay_beta = decay_beta
        self.prior_start = prior_start
        self.prior_floor = prior_floor
        self.dynabo = dynabo

        # Variables to discretize
        self._discretize = discretize
        self._discrete_bins_factor = discrete_bins_factor

    def iteration_number(self, n: int):
        return n - self.prior_start + 1

    def compute_prior(self, X, n):
        # Compute the prior for the configurations
        prior_values = np.ones((len(X), 1))
        # iterate over the hyperparmeters (alphabetically sorted) and the columns, which come
        # in the same order
        for parameter, X_col in zip(self.configspace.values(), X.T):
            prior_values *= parameter._pdf(X_col[:, np.newaxis]) + self.prior_floor
        #if self.iteration_number(n) >= 50:
        #    return np.ones((len(X), 1))
        if self.dynabo:
            return np.power(prior_values, self.decay_beta / np.square(self.iteration_number(n)))
        else:
            return np.power(prior_values, self.decay_beta / self.iteration_number(n))

    def _compute_discretized_pdf(
        self,
        hyperparameter: FloatHyperparameter,
        X_col: np.ndarray,
        number_of_bins: int,
    ) -> np.ndarray:
        """
        Code Taken from PriorAcquisitionFunction._compute_discretized_pdf
        """
        # Evaluates the actual pdf on all the relevant points
        # Replace deprecated method
        pdf_values = hyperparameter._pdf(X_col[:, np.newaxis])

        # Retrieves the largest value of the pdf in the domain
        lower, upper = (0, hyperparameter.get_max_density())

        # Creates the bins (the possible discrete options of the pdf)
        bin_values = np.linspace(lower, upper, number_of_bins)

        # Generates an index (bin) for each evaluated point
        bin_indices = np.clip(np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1).astype(int)

        # Gets the actual value for each point
        prior_values = bin_values[bin_indices]

        return prior_values


class DynamicPriorAcquisitionFunction(PriorAcquisitionFunction):
    """
    Adapt the PriorAcquisitionFunction to enable weighting the prior at any point of the optimization process.

    The existing implemenation is adapted by adding a second configpsace, which contains prior information and adding
    a `dynamic_init` method to set the prior information, just as the intiial prior.
    """

    def __init__(
        self,
        acquisition_function,
        initial_design_size: int,
        dynabo,
        discretize=False,
        discrete_bins_factor=10,
    ):
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._initial_design_size = initial_design_size
        self._dynabo = dynabo
        self._discretize = discretize
        self._discrete_bins_factor = discrete_bins_factor

        self._eta: float | None = None
        self._decay_beta = None
        self._prior_configspaces: List[ConfigSpacePdfWrapper] = list()
        self._prior_floor = None
        self._is_active = False  # The prior is not active at the beginning
        self._current_config_number = None

        # Functions needed by the superclass
        if isinstance(self._acquisition_function, IntegratedAcquisitionFunction):
            acquisition_type = self._acquisition_function._acquisition_function
        else:
            acquisition_type = self._acquisition_function
        self._rescale = isinstance(acquisition_type, (LCB, TS))

        # Tracker for config number
        self.current_config_nuber = None
        self._average_acquisition_function_impact: Dict[str, float] = None
        self._incumbent_acquisition_function_impact: Dict[str, float] = None

    def dynamic_init(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        prior_configspace: ConfigurationSpace,
        decay_beta: float,
        prior_start: int,
        prior_floor: float = 1e-12,
    ):
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self._eta: float | None = None

        self._decay_beta = decay_beta
        self._prior_configspaces.append(
            ConfigSpacePdfWrapper(
                configspace=prior_configspace, decay_beta=decay_beta, prior_start=prior_start, prior_floor=prior_floor, discretize=self._discretize, discrete_bins_factor=self._discrete_bins_factor, dynabo=self._dynabo
            )
        )

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N, 1]
            Prior-weighted acquisition function values of X
        """
        if self._rescale:
            # for TS and UCB, we need to scale the function values to not run into issues
            # of negative values or issues of varying magnitudes (here, they are both)
            # negative by design and just flipping the sign leads to picking the worst point)
            acq_values = np.clip(self._acquisition_function._compute(X) + self._eta, 0, np.inf)
        else:
            acq_values = self._acquisition_function._compute(X)

        # Problem i previously did not consider that we are evaluting 5000 points here
        prior_values = [prior.compute_prior(X, self.current_config_nuber) for prior in self._prior_configspaces]

        if self._prior_configspaces:
            af_impact = np.mean(acq_values)
            prior_impacts = [np.mean(prior_values[i]) for i in range(len(self._prior_configspaces))]
            self._average_acquisition_function_impact = {
                "acquisition_function": af_impact,
                **{f"prior_{i}": prior_impacts[i] for i in range(len(self._prior_configspaces))},
            }

            result = acq_values * np.prod(prior_values, axis=0)
            incumbent_index = np.argmax(result)
            af_impact = acq_values[incumbent_index]
            prior_impacts = [prior_values[i][incumbent_index] for i in range(len(self._prior_configspaces))]
            self._incumbent_acquisition_function_impact = {
                "acquisition_function": af_impact,
                **{f"prior_{i}": prior_impacts[i] for i in range(len(self._prior_configspaces))},
            }
        result = acq_values * np.prod(prior_values, axis=0)
        return result
