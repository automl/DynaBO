from smac.acquisition.function import PriorAcquisitionFunction, AbstractAcquisitionFunction, IntegratedAcquisitionFunction, LCB, TS
from typing import Any
from ConfigSpace import Configuration

class DynamicPriorAcquisitionFunction(PriorAcquisitionFunction):
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        decay_beta: float,
        prior_floor: float = 1e-12,
        discretize: bool = False,
        discrete_bins_factor: float = 10.0,
    ):
        super().__init__()
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self._eta: float | None = None

        self._hyperparameters: dict[Any, Configuration] | None = None
        self._decay_beta = decay_beta
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
        self._iteration_number = 0