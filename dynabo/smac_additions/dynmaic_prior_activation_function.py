from smac.acquisition.function import (
    PriorAcquisitionFunction,
    AbstractAcquisitionFunction,
    IntegratedAcquisitionFunction,
    LCB,
    TS,
)
from typing import Any
from ConfigSpace import Configuration
from smac.model.abstract_model import AbstractModel
from smac.model.random_forest import AbstractRandomForest
from smac.utils.logging import get_logger
from ConfigSpace import ConfigurationSpace

logger = get_logger(__name__)


class DynamicPriorAcquisitionFunction(PriorAcquisitionFunction):
    """
    Adapt the PriorAcquisitionFunction to enable weighting the prior at any point of the optimization process.

    The existing implemenation is adapted by adding a second configpsace, which contains prior information and adding
    a `dynamic_init` method to set the prior information, just as the intiial prior.
    """

    def dynamic_init(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        decay_beta: float,
        prior_configspace: ConfigurationSpace,
        prior_floor: float = 1e-12,
        discretize: bool = False,
        discrete_bins_factor: float = 10.0,
    ):
        super().__init__()
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self._eta: float | None = None

        self._decay_beta = decay_beta
        self._prior_configspace = prior_configspace
        self._hyperparameters: dict[Any, Configuration] | None = dict(self._prior_configspace)
        self._prior_floor = prior_floor
        self._discretize = discretize
        self._discrete_bins_factor = discrete_bins_factor

        # check if the acquisition function is LCB or TS - then the acquisition function values
        # need to be rescaled to assure positiveness & correct magnitude
        if isinstance(self._acquisition_function, IntegratedAcquisitionFunction):
            acquisition_type = self._acquisition_function._acquisition_function
        else:
            acquisition_type = self._acquisition_function

        self._rescale = isinstance(acquisition_type, (LCB, TS))

        # Variables needed to adapt the weighting of the prior
        self._initial_design_size = (
            None  # The amount of datapoints in the initial design
        )
        self._iteration_number = 1  # The amount of configurations the prior was used in the selection of configurations. It starts at 1