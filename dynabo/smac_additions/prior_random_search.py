from smac.acquisition.maximizer import AbstractAcquisitionMaximizer
from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function import AbstractAcquisitionFunction

class PriorRandomSearch(AbstractAcquisitionMaximizer):
    def __init__(self, 
        configspace: ConfigurationSpace,
        prior_configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        ):
        super().__init__(configspace, acquisition_function, challengers=challengers, seed=seed)
        self._prior_configspace = prior_configspace

    def _maximize(self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False):

        if n_points > 1:
            n_prior_random_configs = n_points // 2
            n_rand_configs = n_points - n_prior_random_configs

            rand_configs = self._maximize_random(n_rand_configs)
            prior_random_configs = self._maximize_random(n_prior_random_configs)
        else:
            if self._rng.rand() < 0.5:
                rand_configs = self._maximize_random(1)
                prior_random_configs = []
            else:
                rand_configs = []
                prior_random_configs = self._maximize_random(1)
        

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

    def _maximize_random(
        self,
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        if n_points > 1:
            return  self._prior_configspace.sample_configuration(size=n_points)
        else:
            return [self._prior_configspace.sample_configuration()]
            


