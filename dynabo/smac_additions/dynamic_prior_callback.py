import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
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


class AbstractPriorCallback(Callback, ABC):
    def __init__(
        self,
        scenario: str,
        dataset: str,
        metric: str,
        base_path: str,
        prior_every_n_iterations: int,
        prior_std_denominator: float,
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
        self.prior_std_denominator = prior_std_denominator
        self.initial_design_size = initial_design_size
        self.validate_prior = validate_prior

        self.result_processor = result_processor
        self.evaluator = evaluator

        self.incumbent_performance = float(np.infty)

        self.prior_data_path = self.get_prior_data_path(base_path, scenario, dataset, metric)
        self.prior_data = self.get_prior_data()

        # Number of the prior
        self.prior_number = 0

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
        smbo.intensifier.config_selector._acquisition_function.current_config_nuber = smbo.runhistory.finished

        if self.intervene(smbo):
            prior_configspace, origin_configpsace, performance, logging_config = self.construct_prior(smbo)

            if prior_configspace is not None:
                prior_accepted = self.check_prior(smbo, prior_configspace, origin_configpsace)
                if prior_accepted:
                    self.set_prior(smbo, prior_configspace)
                self.log_prior(smbo=smbo, performance=performance, config=logging_config, prior_accepted=prior_accepted)
            else:
                prior_accepted = False
                self.log_no_prior()

        return super().on_ask_start(smbo)

    def check_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace, origin_configpsace: ConfigurationSpace) -> bool:
        if self.validate_prior:
            # Sample configurations according to both the prior and the standard configpsace
            # prior_configs = prior_configspace.sample_configuration(1000)
            # origin_configs = origin_configpsace.sample_configuration(1000)

            # mannwhitneyu()

            # Do Man Whitney U Test, to determien whether or not we use the prior
            raise NotImplementedError("Please implement the validate_prior method.")
        return True

    @abstractmethod
    def intervene(self, smbo: SMBO) -> bool:
        pass

    def construct_prior(self, smbo: SMBO) -> Tuple[ConfigurationSpace, ConfigurationSpace, float, Dict]:
        """
        Sets a new prior on the acquisition function and configspace.
        """
        self.prior_number += 1

        current_incumbent = smbo.intensifier.get_incumbent()
        incumbent_performance = (-1) * smbo.runhistory.get_cost(current_incumbent)

        sampled_config = self.sample_prior(smbo, incumbent_performance)

        if sampled_config is None:
            return None, None, None, None

        performance = sampled_config["score"].values[0]
        hyperparameter_config = sampled_config[[col for col in sampled_config.columns if col.startswith("config_")]]
        hyperparameter_config.columns = [col.replace("config_", "") for col in hyperparameter_config.columns]
        hyperparameter_config = hyperparameter_config.iloc[0].to_dict()

        origin_configspace = smbo._configspace
        prior_configspace = build_prior_configuration_space(origin_configspace, hyperparameter_config, prior_std_denominator=self.prior_std_denominator * self.prior_number)

        return prior_configspace, origin_configspace, performance, hyperparameter_config

    def set_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace):
        smbo.intensifier.config_selector._acquisition_maximizer.dynamic_init(prior_configspace)
        smbo.intensifier.config_selector._acquisition_function.dynamic_init(
            acquisition_function=smbo.intensifier.config_selector._acquisition_function._acquisition_function,
            prior_configspace=prior_configspace,
            decay_beta=smbo._scenario.n_trials / 10,
            prior_start=smbo.runhistory.finished,
        )

    @abstractmethod
    def sample_prior(self, smbo: SMBO, incumbent_performance: float) -> pd.DataFrame:
        """
        Samples a prior from the prior data.
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
                        "no_superior_configuration": False,
                        "performance": performance,
                        "configuration": str(config),
                        "after_n_evaluations": smbo.runhistory.finished,
                        "after_runtime": self.evaluator.accumulated_runtime,
                        "after_virtual_runtime": self.evaluator.accumulated_runtime + self.evaluator.reasoning_runtime,
                        "after_reasoning_runtime": self.evaluator.reasoning_runtime,
                    }
                }
            )

    def log_no_prior(self):
        if self.result_processor is not None:
            self.result_processor.process_logs(
                {
                    "priors": {
                        "no_superior_configuration": True,
                    }
                }
            )
    
class DynaBOAbstractPriorCallback(AbstractPriorCallback):
    def intervene(self, smbo: SMBO) -> bool:
        return smbo.runhistory.finished >= self.initial_design_size and (smbo.runhistory.finished - self.initial_design_size) % self.prior_every_n_iterations == 0


class PiBOAbstractCallback(DynaBOAbstractPriorCallback):
    def intervene(self, smbo):
        return smbo.runhistory.finished == self.initial_design_size and (smbo.runhistory.finished - self.initial_design_size) % self.prior_every_n_iterations == 0


class DynaBOWellPerformingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> Optional[pd.DataFrame]:
        # Select all configurations that have a better performance than the incumbent
        # Chekc whether the valeus are sorted
        better_performing_configs = self.prior_data[self.prior_data["score"] > incumbent_performance]
        

        if better_performing_configs.empty:
            return None
        n = len(better_performing_configs)
        
        # Generate exponential samples and map them to valid indices
        raw_samples = smbo.intensifier._rng.exponential(scale = 5, size=1)
        indices = np.round(raw_samples).astype(int)
        indices = np.clip(indices, 0, n - 1)  # Ensure indices are within bounds

        return better_performing_configs.iloc[indices]

class PiBOWellPerformingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> Optional[pd.DataFrame]:
        # Select all configurations that have a better performance than the incumbent
        # Chekc whether the valeus are sorted
        better_performing_configs = self.prior_data[self.prior_data["score"] > incumbent_performance]
        

        if better_performing_configs.empty:
            return None
        n = len(better_performing_configs)
        
        # Generate exponential samples and map them to valid indices
        raw_samples = smbo.intensifier._rng.exponential(scale = 5, size=1)
        indices = np.round(raw_samples).astype(int)
        indices = np.clip(indices, 0, n - 1)  # Ensure indices are within bounds

        return better_performing_configs.iloc[indices]

class DynaBOMediumPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        sampled_config = self.prior_data.sample(random_state=smbo.intensifier._rng)
        return sampled_config
    
class PiBOMediumPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        sampled_config = self.prior_data.sample(random_state=smbo.intensifier._rng)
        return sampled_config


class DynaBOMissleadingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        better_performing_configs = self.prior_data[self.prior_data["score"] < incumbent_performance]

        # Sample from the considered configurations
        sampled_config = better_performing_configs.sample(random_state=smbo.intensifier._rng)
        return sampled_config
    
class PiBOMissleadingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        better_performing_configs = self.prior_data[self.prior_data["score"] < incumbent_performance]

        # Sample from the considered configurations
        sampled_config = better_performing_configs.sample(random_state=smbo.intensifier._rng)
        return sampled_config


class DynaBODeceivingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        raise NotImplementedError("Please implement the sample_prior method.")

class PiBODeceivingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        raise NotImplementedError("Please implement the sample_prior method.")