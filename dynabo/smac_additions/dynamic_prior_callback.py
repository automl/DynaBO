import os
from abc import ABC, abstractmethod

import pandas as pd
from smac.callback import Callback
from smac.main.smbo import SMBO

from dynabo.utils.configspace_utils import build_prior_configuration_space


class AbstractDynamicPriorCallback(Callback, ABC):
    def __init__(self, scenario: str, dataset: str, metric: str, base_path: str, prior_every_n_iterations: int, initial_design_size: int):
        super().__init__()
        self.scenario = scenario
        self.dataset = dataset
        self.metric = metric
        self.base_path = base_path
        self.prior_every_n_iterations = prior_every_n_iterations
        self.initial_design_size = initial_design_size

        self.prior_data_path = self.get_prior_data_path(base_path, scenario, dataset, metric)
        self.prior_data = self.get_prior_data()

    @staticmethod
    def get_prior_data_path(base_path, scenario: str, dataset: str, metric: str) -> str:
        """
        Returns the path to the prior data.
        """
        return os.path.join(base_path, scenario, dataset, metric, "prior_table.csv")

    def get_prior_data(
        self,
    ) -> pd.DataFrame:
        return pd.read_csv(self.prior_data_path)

    def on_iteration_start(self, smbo: SMBO):
        "We add prior information, before the next iteration is started."

        if self.intervene(smbo):
            self.set_prior(smbo)
        return super().on_iteration_start(smbo)

    def intervene(self, smbo: SMBO) -> bool:
        return smbo.runhistory.finished >= self.initial_design_size and smbo.runhistory.finished % self.prior_every_n_iterations == 0

    @abstractmethod
    def set_prior(self, smbo: SMBO):
        """
        Sets a new prior on the acquisition function and configspace.
        """
        # new_prior_configspace = self.intervention_schedule[len(smbo.runhistory)]

        # acquisition_function_maximizer: LocalAndPriorSearch = smbo.intensifier.config_selector._acquisition_maximizer
        # acquisition_function_maximizer.dynamic_init(new_prior_configspace)

        # acquisition_function: DynamicPriorAcquisitionFunction = smbo.intensifier.config_selector._acquisition_function
        # acquisition_function.dynamic_init(
        #    acquisition_function=acquisition_function._acquisition_function,
        #    prior_configspace=new_prior_configspace,
        #    decay_beta=smbo._scenario.n_trials / 10,
        # )


class WellPerformingPriorCallback(AbstractDynamicPriorCallback):
    def set_prior(self, smbo: SMBO):
        current_incumbent = smbo.intensifier.get_incumbent()
        incumbent_performance = (-1) * smbo.runhistory.get_cost(current_incumbent)

        # Select all configurations that have a better performance than the incumbent
        better_performing_configs = self.prior_data[self.prior_data["score"] > incumbent_performance]

        # Sample from the considered configurations
        sampled_config = better_performing_configs.sample(random_state=smbo.intensifier._rng)
        hyperparameter_config = sampled_config[[col for col in sampled_config.columns if col.startswith("config_")]]
        hyperparameter_config.columns = [col.replace("config_", "") for col in hyperparameter_config.columns]
        hyperparameter_config = hyperparameter_config.iloc[0].to_dict()

        origin_configspace = smbo._configspace
        prior_configspace = build_prior_configuration_space(origin_configspace, hyperparameter_config)
        smbo.intensifier.config_selector._acquisition_maximizer.dynamic_init(prior_configspace)
        smbo.intensifier.config_selector._acquisition_function.dynamic_init(
            acquisition_function=smbo.intensifier.config_selector._acquisition_function._acquisition_function,
            prior_configspace=prior_configspace,
            decay_beta=smbo._scenario.n_trials / 10,
            prior_start=smbo.runhistory.finished,
        )
