from typing import Callable, List

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer import AbstractAcquisitionMaximizer


class PriorConfigSpaceWrapper:
    def __init__(self, configspace: ConfigurationSpace, prior_decayy: Callable[[Configuration], float]):
        self._configspace = configspace
        self._importance_function = prior_decayy
        self._n_configurations_sampled = 0

    def percentage(self) -> float:
        importance = self._importance_function(self._n_configurations_sampled)
        self._n_configurations_sampled += 1
        return importance

    def sample_configuration(self, size: int) -> List[Configuration]:
        return self._configspace.sample_configuration(size=size)


class PriorRandomSearch(AbstractAcquisitionMaximizer):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
    ):
        super().__init__(configspace, acquisition_function, challengers=challengers, seed=seed)
        self._prior_configspace: List[PriorConfigSpaceWrapper] = list()
        self._is_active: bool = False
        self._n_active_configs: int = None
        self._prior_decay: Callable[[float], float] = None

    def _dynamic_init(
        self,
        prior_configspace: ConfigurationSpace,
        prior_decay: Callable[[float], float] = lambda x: np.exp(-0.126 * x),
    ):
        decayed_prior_configspace = PriorConfigSpaceWrapper(prior_configspace, prior_decay)
        self._prior_configspace.append(decayed_prior_configspace)

        self._is_active = True

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False,
    ):
        if not self._is_active:
            configs = self._maximize_random(n_points)
        else:
            importances = np.array([prior.percentage() for prior in self._prior_configspace])
            sum_of_importances = np.sum(importances)

            # If we want to sample more than 90% accoridng to the priors
            if sum_of_importances > 0.9:
                # Normalize the importances against each other
                normalized_importances = importances / sum_of_importances

                # Only assign 90% of the points to the prior
                normalized_importances = normalized_importances * 0.9

                # Assign the rest to random search
                rand_importance = 1 - np.sum(normalized_importances)

            else:
                normalized_importances = importances
                rand_importance = 1 - np.sum(normalized_importances)

            # If we sample more than 1 point
            if n_points > 1:
                configs = list()
                for i in range(len(normalized_importances)):
                    n_samples = int(np.round(n_points * normalized_importances[i]))

                    if n_samples > 0:
                        configs += self._maximize_prior(n_samples, self._prior_configspace[i])

                n_samples = int(np.round(n_points * rand_importance))
                configs += self._maximize_random(n_samples)

            # If we sample one point
            else:
                # convert pdf to cdf
                cdf = np.cumsum(normalized_importances)

                # draw a random number
                rand = np.random.rand()

                if any(cdf > rand):
                    index = np.argmax(cdf > rand)
                    configs = self._maximize_prior(1, self._prior_configspace[index])

                else:
                    configs = self._maximize_random(1)

        if _sorted:
            for i in range(len(configs)):
                configs[i].origin = "Acquisition Function Maximizer: Prior Random Search"

            return self._sort_by_acquisition_value(configs)
        else:
            for i in range(len(configs)):
                configs[i].origin = "Acquisition Function Maximizer: Prior Random Search"

            return [(0, configs[i]) for i in range(len(configs))]

    def _maximize_random(
        self,
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        if n_points > 1:
            return self._configspace.sample_configuration(size=n_points)
        else:
            return [self._configspace.sample_configuration()]

    def _maximize_prior(
        self,
        n_points: int,
        configspace: ConfigurationSpace,
    ) -> list[tuple[float, Configuration]]:
        if n_points > 1:
            return configspace.sample_configuration(size=n_points)
        else:
            return [configspace.sample_configuration(n_points)]
