from smac.acquisition.maximizer import AbstractAcquisitionMaximizer
from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function import AbstractAcquisitionFunction
from typing import Callable
import numpy as np

class PriorRandomSearch(AbstractAcquisitionMaximizer):
    def __init__(self, 
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        ):
        super().__init__(configspace, acquisition_function, challengers=challengers, seed=seed)
        self._prior_configspace = None
        self._is_active:bool = False
        self._n_active_configs: int = None
        self._prior_decay: Callable[[float], float] = None

    def _dynamic_init(self, prior_configspace: ConfigurationSpace, prior_decay: Callable[[float], float] = lambda x: 0.3 * np.exp(-0.126 * x)) :
        self._prior_configspace = prior_configspace
        self._is_active = True
        self._n_active_configs = None
        self._prior_decay = prior_decay

    def _maximize(self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False):

        if self._is_active and self._n_active_configs is None:
            self._n_active_configs = len(previous_configs)

        if n_points > 1:
            if self._is_active:
                
                n_prior_random_configs = round(self._prior_decay(len(previous_configs) - self._n_active_configs) * n_points)
                n_rand_configs = n_points - n_prior_random_configs

                rand_configs = self._maximize_random(n_rand_configs)
                prior_random_configs = self._maximize_prior(n_prior_random_configs)
            else:
                rand_configs = self._maximize_random(n_points)
                prior_random_configs = []
        else:
            if self._is_active:
                if self._rng.rand() < self._prior_decay(len(previous_configs) - self._n_active_configs):
                    rand_configs = self._maximize_random(1)
                    prior_random_configs = []
                else:
                    rand_configs = []
                    prior_random_configs = self._maximize_prior(1)
            else:
                rand_configs = self._maximize_random(1)
                prior_random_configs = []
        

        configs = rand_configs + prior_random_configs

        if _sorted:
            for i in range(len(configs)):
                configs[i].origin = "Acquisition Function Maximizer: Random Search (sorted)"

            return self._sort_by_acquisition_value(configs)
        else:
            for i in range(len(configs)):
                configs[i].origin = "Acquisition Function Maximizer: Random Search"

            return [(0, configs[i]) for i in range(len(configs))]

    def _maximize_random(
        self,
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        if n_points > 1:
            return  self._configspace.sample_configuration(size=n_points)
        else:
            return [self._configspace.sample_configuration()]

    def _maximize_prior(
        self,
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        if n_points > 1:
            return  self._prior_configspace.sample_configuration(size=n_points)
        else:
            return [self._prior_configspace.sample_configuration()]
            


