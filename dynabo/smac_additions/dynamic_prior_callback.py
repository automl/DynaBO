import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from py_experimenter.result_processor import ResultProcessor
from scipy.stats import mannwhitneyu
from smac.acquisition.function import LCB
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue

from dynabo.smac_additions.dynmaic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.utils.configspace_utils import build_prior_configuration_space
from dynabo.utils.evaluator import YAHPOGymEvaluator

PERFORMANCE_INDICATOR_COLUMN = "performance"


class PriorValidationMethod(Enum):
    MANN_WHITNEY_U = "mann_whitney_u"
    DIFFERENCE = "difference"


class LogIncumbentCallback(Callback):
    def __init__(self, result_processor: ResultProcessor, evaluator: YAHPOGymEvaluator):
        super().__init__()
        self.result_processor = result_processor
        self.evaluator = evaluator
        self.incumbent_performance = float(np.infty)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        if (
            isinstance(smbo.intensifier.config_selector._acquisition_function, DynamicPriorAcquisitionFunction)
            and smbo.intensifier.config_selector._acquisition_function._average_acquisition_function_impact is not None
        ):
            average_acquisition_function_impact = str(smbo.intensifier.config_selector._acquisition_function._average_acquisition_function_impact)
            incumbent_acquisition_function_impact = str(smbo.intensifier.config_selector._acquisition_function._incumbent_acquisition_function_impact)
        else:
            average_acquisition_function_impact = None
            incumbent_acquisition_function_impact = None

        if self.result_processor is not None:
            if value.cost < self.incumbent_performance:
                self.incumbent_performance = value.cost
                incumbent = True
            else:
                incumbent = False

            self.result_processor.process_logs(
                {
                    "configs": {
                        "performance": (-1) * value.cost,
                        "incumbent": incumbent,
                        "configuration": str(dict(info.config)),
                        "after_n_evaluations": smbo._runhistory.finished,
                        "after_runtime": self.evaluator.accumulated_runtime,
                        "after_virtual_runtime": self.evaluator.accumulated_runtime + self.evaluator.reasoning_runtime,
                        "after_reasoning_runtime": self.evaluator.reasoning_runtime,
                        "average_acquisition_function_impact": average_acquisition_function_impact,
                        "incumbent_acquisition_function_impact": incumbent_acquisition_function_impact,
                    }
                }
            )


class AbstractPriorCallback(Callback, ABC):
    def __init__(
        self,
        benchmarklib,
        scenario: str,
        dataset: str,
        metric: str,
        base_path: str,
        initial_design_size: int,
        prior_every_n_trials: int,
        validate_prior: bool,
        prior_validation_method: PriorValidationMethod,
        n_prior_validation_samples: int,
        prior_validation_manwhitney_p_value: float,
        prior_validation_difference_threshold: float,
        prior_std_denominator: float,
        prior_decay_enumerator: int,
        prior_decay_denominator: int,
        exponential_prior: bool,
        prior_sampling_weight: float,
        result_processor: ResultProcessor = None,
        evaluator: YAHPOGymEvaluator = None,
    ):
        super().__init__()
        self.benchmarklib = benchmarklib
        self.scenario = scenario
        self.dataset = dataset
        self.metric = metric
        self.base_path = base_path
        self.initial_design_size = initial_design_size
        self.prior_every_n_trials = prior_every_n_trials
        self.validate_prior = validate_prior
        self.prior_validation_method = prior_validation_method
        self.prior_validation_manwhitney_p = prior_validation_manwhitney_p_value
        self.prior_validation_difference_threshold = prior_validation_difference_threshold
        self.n_prior_validation_samples = n_prior_validation_samples
        self.prior_std_denominator = prior_std_denominator
        self.prior_decay_enumerator = prior_decay_enumerator
        self.prior_decay_denominator = prior_decay_denominator

        self.exponential_prior = exponential_prior
        self.prior_sampling_weight = prior_sampling_weight

        self.result_processor = result_processor
        self.evaluator = evaluator

        self.incumbent_performance = float(np.infty)

        if self.validate_prior:
            self.lcb = LCB()

        self.prior_data_path = self.get_prior_data_path(base_path, self.benchmarklib, scenario, dataset, metric)
        self.prior_data = self.get_prior_data()

        # Number of the prior
        self.prior_number = 0

    @staticmethod
    def get_prior_data_path(base_path, benchmarklib, scenario: str, dataset: str, metric: str) -> str:
        """
        Returns the path to the prior data.
        """
        # TODO adapt
        path = os.path.join(base_path, benchmarklib, scenario)
        if benchmarklib == "yahoo":
            path = os.path.join(path, dataset, metric)
        path = os.path.join(path, "prior_table.csv")
        return path

    def get_prior_data(
        self,
    ) -> pd.DataFrame:
        return pd.read_csv(self.prior_data_path)

    def on_ask_start(self, smbo: SMBO):
        "We add prior information, before the next iteration is started."
        smbo.intensifier.config_selector._acquisition_function.current_config_nuber = smbo.runhistory.finished

        if self.intervene(smbo):
            prior_configspace, origin_configpsace, performance, logging_config, superior_configuraiton = self.construct_prior(smbo)

            prior_accepted, prior_mean_acq_value, origin_mean_acq_value = self.accept_prior(smbo, prior_configspace, origin_configpsace)

            if prior_accepted:
                self.set_prior(smbo, prior_configspace)
            self.log_prior(
                smbo=smbo,
                performance=performance,
                config=logging_config,
                prior_accepted=prior_accepted,
                prior_mean_acq_value=prior_mean_acq_value,
                origin_mean_acq_value=origin_mean_acq_value,
                superior_configuration=superior_configuraiton,
            )

        return super().on_ask_start(smbo)

    def accept_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace, origin_configspace: ConfigurationSpace) -> bool:
        if self.validate_prior:
            current_incumbent = smbo.intensifier.get_incumbent()
            incumbent_configuration_dict = dict(current_incumbent)
            incumbent_configuration_space = build_prior_configuration_space(origin_configspace, incumbent_configuration_dict, prior_std_denominator=self.prior_std_denominator * self.prior_number)

            prior_samples = prior_configspace.sample_configuration(size=self.n_prior_validation_samples)
            incumbent_samples = incumbent_configuration_space.sample_configuration(size=self.n_prior_validation_samples)

            self.lcb.update(model=smbo.intensifier.config_selector._acquisition_function.model, num_data=smbo.runhistory.finished)
            lcb_prior_values = self.lcb(prior_samples).squeeze()
            lcb_incumbent_values = self.lcb(incumbent_samples).squeeze()
            if self.prior_validation_method == PriorValidationMethod.MANN_WHITNEY_U.value:
                result = mannwhitneyu(
                    lcb_prior_values,
                    lcb_incumbent_values,
                    alternative="less",
                )

                if result.pvalue < self.prior_validation_manwhitney_p:
                    return False, lcb_prior_values.mean(), lcb_incumbent_values.mean()
                else:
                    return True, lcb_prior_values.mean(), lcb_incumbent_values.mean()
            elif self.prior_validation_method == PriorValidationMethod.DIFFERENCE.value:
                result = lcb_prior_values.mean() - lcb_incumbent_values.mean()
                if result > self.prior_validation_difference_threshold:
                    return True, lcb_prior_values.mean(), lcb_incumbent_values.mean()
                else:
                    return False, lcb_prior_values.mean(), lcb_incumbent_values.mean()
            else:
                raise ValueError(f"Prior validation method {self.prior_validation_method} not supported.")

        return True, None, None

    @abstractmethod
    def intervene(self, smbo: SMBO) -> bool:
        pass

    def construct_prior(self, smbo: SMBO) -> Tuple[ConfigurationSpace, ConfigurationSpace, float, Dict, bool]:
        """
        Sets a new prior on the acquisition function and configspace.
        """
        self.prior_number += 1

        current_incumbent = smbo.intensifier.get_incumbent()
        incumbent_performance = (-1) * smbo.runhistory.get_cost(current_incumbent)

        sampled_config = self.sample_prior(smbo, incumbent_performance)
        prior_performance = sampled_config[PERFORMANCE_INDICATOR_COLUMN].values[0]

        # Check if the sampled configuration is better than the incumbent
        superior_configuration = bool(prior_performance > incumbent_performance)

        if sampled_config is None:
            raise ValueError("No prior configuration could be sampled.")

        performance = sampled_config[PERFORMANCE_INDICATOR_COLUMN].values[0]
        hyperparameter_config = sampled_config[[col for col in sampled_config.columns if col.startswith("config_")]]
        hyperparameter_config.columns = [col.replace("config_", "") for col in hyperparameter_config.columns]
        hyperparameter_config = hyperparameter_config.iloc[0].to_dict()

        origin_configspace = smbo._configspace
        prior_configspace = build_prior_configuration_space(origin_configspace, hyperparameter_config, prior_std_denominator=self.prior_std_denominator * self.prior_number)

        return prior_configspace, origin_configspace, performance, hyperparameter_config, superior_configuration

    def set_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace):
        smbo.intensifier.config_selector._acquisition_maximizer.dynamic_init(prior_configspace)
        smbo.intensifier.config_selector._acquisition_function.dynamic_init(
            acquisition_function=smbo.intensifier.config_selector._acquisition_function._acquisition_function,
            prior_configspace=prior_configspace,
            decay_beta=self.prior_decay_enumerator / self.prior_decay_denominator,
            prior_start=smbo.runhistory.finished,
        )

    @abstractmethod
    def sample_prior(self, smbo: SMBO, incumbent_performance: float) -> pd.DataFrame:
        """
        Samples a prior from the prior data.
        """

    def log_prior(self, smbo: SMBO, performance: float, config: Dict, prior_accepted: bool, prior_mean_acq_value: float, origin_mean_acq_value: float, superior_configuration: bool):
        """
        Logs the prior data.
        """
        if self.result_processor is not None:
            self.result_processor.process_logs(
                {
                    "priors": {
                        "prior_accepted": prior_accepted,
                        "superior_configuration": superior_configuration,
                        "performance": performance,
                        "prior_mean_acq_value": prior_mean_acq_value,
                        "origin_mean_acq_value": origin_mean_acq_value,
                        "configuration": str(config),
                        "after_n_evaluations": smbo.runhistory.finished,
                        "after_runtime": self.evaluator.accumulated_runtime,
                        "after_virtual_runtime": self.evaluator.accumulated_runtime + self.evaluator.reasoning_runtime,
                        "after_reasoning_runtime": self.evaluator.reasoning_runtime,
                    }
                }
            )

    def draw_sample(self, df: pd.DataFrame, rng) -> np.ndarray:
        """
        Draws an exponential sample from the data. Samples with smaller indices are more likely to be drawn.
        """
        if df.shape[0] == 0:
            return None

        if self.exponential_prior:
            weights = np.exp(-self.prior_sampling_weight * df.index).values
            weights_normalized = weights / weights.sum()  # Normalize weights

            sample = df.sample(weights=weights_normalized, random_state=rng)
        else:
            sample = df.sample(random_state=rng)
        return sample


class DynaBOAbstractPriorCallback(AbstractPriorCallback):
    def intervene(self, smbo: SMBO) -> bool:
        # To use the surrogate, we need to sample one additional config here
        return smbo.runhistory.finished >= self.initial_design_size + 1 and (smbo.runhistory.finished - self.initial_design_size - 1) % self.prior_every_n_trials == 0


class PiBOAbstractPriorCallback(AbstractPriorCallback):
    def intervene(self, smbo):
        # To use the surrogate, we need to sample one additional config here
        return smbo.runhistory.finished == self.initial_design_size + 1 and (smbo.runhistory.finished - self.initial_design_size - 1) % self.prior_every_n_trials == 0

    def accept_prior(self, smbo, prior_configspace, origin_configspace):
        return True, None, None


class DynaBOWellPerformingPriorCallback(DynaBOAbstractPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile

    def sample_prior(self, smbo, incumbent_performance) -> Optional[pd.DataFrame]:
        # Select all configurations that have a better performance than the incumbent
        # Check whether the values are sorted
        if (self.prior_data[PERFORMANCE_INDICATOR_COLUMN] >= incumbent_performance).any():
            relevant_configs: pd.DataFrame = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] >= incumbent_performance]
            relevant_configs = relevant_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN)
        else:
            relevant_configs = self.prior_data.sort_values(PERFORMANCE_INDICATOR_COLUMN)
            # If not better performing configurations are available, sample from no_incumbent_percentile of the configurations
            relevant_configs = relevant_configs.head(int(self.no_incumbent_percentile * relevant_configs.shape[0]))

        return self.draw_sample(relevant_configs, smbo.intensifier._rng)


class PiBOWellPerformingPriorCallback(PiBOAbstractPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile

    def sample_prior(self, smbo, incumbent_performance) -> Optional[pd.DataFrame]:
        # Select all configurations that have a better performance than the incumbent
        # Chekc whether the valeus are sorted
        if (self.prior_data[PERFORMANCE_INDICATOR_COLUMN] >= incumbent_performance).any():
            relevant_configs: pd.DataFrame = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] >= incumbent_performance]
            relevant_configs = relevant_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN)
        else:
            relevant_configs = self.prior_data.sort_values(PERFORMANCE_INDICATOR_COLUMN)
            # If not better performing configurations are available, sample from no_incumbent_percentile of the configurations
            relevant_configs = relevant_configs.head(int(self.no_incumbent_percentile * relevant_configs.shape[0]))

        return self.draw_sample(relevant_configs, smbo.intensifier._rng)


class DynaBOMediumPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # 50 percent likelihood sample better, with 50% likelihood sample worse
        if smbo.intensifier._rng.random() < 0.5:
            better_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] > incumbent_performance]
            better_performing_configs = better_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=True)
        else:
            better_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] < incumbent_performance]
            better_performing_configs = better_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=False)

        return self.draw_sample(better_performing_configs, smbo.intensifier._rng)


class PiBOMediumPriorCallback(PiBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        # TODO with 50 percent likelihood sample better, with 50% likelihood sample worse. We use the same distribution
        if smbo.intensifier._rng.random() < 0.5:
            better_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] > incumbent_performance]
            better_performing_configs = better_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=True)
        else:
            better_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] < incumbent_performance]
            better_performing_configs = better_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=False)

        return self.draw_sample(better_performing_configs, smbo.intensifier._rng)


class DynaBOMisleadingPriorCallback(DynaBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a worse performance than the incumbent
        worse_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] < incumbent_performance]
        worse_performing_configs = worse_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=False)

        # Sample from the considered configurations
        return self.draw_sample(worse_performing_configs, smbo.intensifier._rng)


class PiBOMisleadingPriorCallback(PiBOAbstractPriorCallback):
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        # Select all configurations that have a better performance than the incumbent
        worse_performing_configs = self.prior_data[self.prior_data[PERFORMANCE_INDICATOR_COLUMN] < incumbent_performance]
        worse_performing_configs = worse_performing_configs.sort_values(PERFORMANCE_INDICATOR_COLUMN, ascending=False)

        # Sample from the considered configurations
        return self.draw_sample(worse_performing_configs, smbo.intensifier._rng)


class DynaBODeceivingPriorCallback(DynaBOAbstractPriorCallback):
    # Negative optimization
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        raise NotImplementedError("Please implement the sample_prior method.")


class PiBODeceivingPriorCallback(DynaBOAbstractPriorCallback):
    # TODO engative
    def sample_prior(self, smbo, incumbent_performance) -> pd.DataFrame:
        raise NotImplementedError("Please implement the sample_prior method.")
