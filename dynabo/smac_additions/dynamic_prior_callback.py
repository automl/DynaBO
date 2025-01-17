import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from py_experimenter.result_processor import ResultProcessor
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue

from dynabo.utils.configspace_utils import build_prior_configuration_space
from dynabo.utils.yahpogym_evaluator import YAHPOGymEvaluator


class LogIncumbentCallback(Callback):
    def __init__(self, result_processor: ResultProcessor, evaluator: YAHPOGymEvaluator):
        super().__init__()
        self.result_processor = result_processor
        self.evaluator = evaluator
        self.incumbent_performance = float(np.infty)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        if self.result_processor is not None:
            if value.cost < self.incumbent_performance:
                self.incumbent_performance = value.cost

                self.result_processor.process_logs(
                    {
                        "incumbents": {
                            "performance": (-1) * value.cost,
                            "configuration": str(dict(info.config)),
                            "after_n_evaluations": smbo._runhistory.finished,
                            "after_runtime": self.evaluator.accumulated_runtime,
                            "after_virtual_runtime": self.evaluator.accumulated_runtime + self.evaluator.reasoning_runtime,
                            "after_reasoning_runtime": self.evaluator.reasoning_runtime,
                        }
                    }
                )


class AbstractDynamicPriorCallback(Callback, ABC):
    def __init__(
        self,
        scenario: str,
        dataset: str,
        metric: str,
        base_path: str,
        prior_every_n_iterations: int,
        initial_design_size: int,
        validate_prior: bool = False,
        result_processor: ResultProcessor = None,
        evaluator: YAHPOGymEvaluator = None,
    ):
        super().__init__()
        self.scenario = scenario
        self.dataset = dataset
        self.metric = metric
        self.base_path = base_path
        self.prior_every_n_iterations = prior_every_n_iterations
        self.initial_design_size = initial_design_size
        self.validate_prior = validate_prior

        self.result_processor = result_processor
        self.evaluator = evaluator

        self.incumbent_performance = float(np.infty)

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

    def on_ask_start(self, smbo: SMBO):
        "We add prior information, before the next iteration is started."

        if self.intervene(smbo):
            prior_accepted = self.check_prior()
            performance, logging_config = self.set_prior(smbo)
            self.log_prior(smbo=smbo, performance=performance, config=logging_config, prior_accepted=prior_accepted)
        return super().on_ask_start(smbo)

    def check_prior(self) -> bool:
        if self.validate_prior:
            raise NotImplementedError("Please implement the validate_prior method.")
        return True

    def intervene(self, smbo: SMBO) -> bool:
        return smbo.runhistory.finished >= self.initial_design_size and smbo.runhistory.finished % self.prior_every_n_iterations == 0

    @abstractmethod
    def set_prior(self, smbo: SMBO):
        """
        Sets a new prior on the acquisition function and configspace.
        """

    def log_prior(self, smbo: SMBO, performance: float, config: Dict, prior_accepted: bool):
        """
        Logs the prior data.
        """
        if self.result_processor is not None:
            self.result_processor.process_logs(
                {
                    "priors": {
                        "prior_accepted": prior_accepted,
                        "performance": performance,
                        "configuration": str(config),
                        "after_n_evaluations": smbo.runhistory.finished,
                        "after_runtime": self.evaluator.accumulated_runtime,
                        "after_virtual_runtime": self.evaluator.accumulated_runtime + self.evaluator.reasoning_runtime,
                        "after_reasoning_runtime": self.evaluator.reasoning_runtime,
                    }
                }
            )


class WellPerformingPriorCallback(AbstractDynamicPriorCallback):
    def set_prior(self, smbo: SMBO):
        current_incumbent = smbo.intensifier.get_incumbent()
        incumbent_performance = (-1) * smbo.runhistory.get_cost(current_incumbent)

        # Select all configurations that have a better performance than the incumbent
        better_performing_configs = self.prior_data[self.prior_data["score"] > incumbent_performance]

        # Sample from the considered configurations
        sampled_config = better_performing_configs.sample(random_state=smbo.intensifier._rng)
        performance = sampled_config["score"].values[0]
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

        return performance, hyperparameter_config
