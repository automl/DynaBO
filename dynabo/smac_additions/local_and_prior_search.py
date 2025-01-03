from __future__ import annotations

from typing import Any, Callable

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.acquisition.maximizer.local_search import LocalSearch
from smac.utils.logging import get_logger

from dynabo.smac_additions.prior_random_search import PriorRandomSearch

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class LocalAndPriorSearch(AbstractAcquisitionMaximizer):
    """
    Implement SMAC's default acquisition function optimization adapted to use PriorSearch.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        seed: int = 0,
        uniform_configspace: ConfigurationSpace | None = None,
        prior_sampling_fraction: float | None = None,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )
        self._random_search = PriorRandomSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            seed=seed,
        )

        self._local_search = LocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )

        self._local_search_iterations = local_search_iterations
        self._prior_sampling_fraction = prior_sampling_fraction
        self._uniform_configspace = uniform_configspace

    def dynamic_init(
        self,
        prior_configspace: ConfigurationSpace,
        prior_decay: Callable[[float], float] = lambda x: np.exp(-0.126 * x),
    ):
        self._random_search._dynamic_init(prior_configspace, prior_decay)

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "random_search": self._random_search.meta,
                "local_search": self._local_search.meta,
            }
        )

        return meta

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        # Get configurations sorted by acquisition function value
        next_configs_by_random_search_sorted = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=n_points,
            _sorted=True,
        )

        # Choose the best self._local_search_iterations random configs to start the local search, and choose only
        # incumbent from previous configs
        random_starting_points = next_configs_by_random_search_sorted[: self._local_search_iterations]
        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=random_starting_points,
        )

        next_configs_by_acq_value = next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        logger.debug(f"First 5 acquisition function values of selected configurations: \n{', '.join(first_five)}")

        return next_configs_by_acq_value
