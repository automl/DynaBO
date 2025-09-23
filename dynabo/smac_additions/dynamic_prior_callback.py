import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple
import gower
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter
from py_experimenter.result_processor import ResultProcessor
from scipy.stats import mannwhitneyu
from smac.acquisition.function import LCB
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue

from dynabo.smac_additions.dynmaic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.utils.configspace_utils import build_prior_configuration_space
from dynabo.utils.evaluator import YAHPOGymEvaluator
from copy import deepcopy

COST_INDICATOR_COLUMN = "cost"


class PriorValidationMethod(Enum):
    MANN_WHITNEY_U = "mann_whitney_u"
    DIFFERENCE = "difference"
    BASELINE_PERFECT = "baseline_perfect"


class LogIncumbentCallback(Callback):
    def __init__(self, result_processor: ResultProcessor, evaluator: YAHPOGymEvaluator, invert_cost: bool = False):
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
                        "cost": value.cost,
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
        validate_prior: bool,
        prior_validation_method: PriorValidationMethod,
        n_prior_validation_samples: int,
        n_prior_based_samples: int,
        prior_validation_manwhitney_p_value: float,
        prior_validation_difference_threshold: float,
        prior_std_denominator: float,
        prior_decay_enumerator: int,
        prior_decay_denominator: int,
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
        self.validate_prior = validate_prior
        self.prior_validation_method = prior_validation_method
        self.prior_validation_manwhitney_p = prior_validation_manwhitney_p_value
        self.prior_validation_difference_threshold = prior_validation_difference_threshold
        self.n_prior_validation_samples = n_prior_validation_samples
        self.n_prior_based_samples = n_prior_based_samples
        self.prior_std_denominator = prior_std_denominator
        self.prior_decay_enumerator = prior_decay_enumerator
        self.prior_decay_denominator = prior_decay_denominator

        self.result_processor = result_processor
        self.evaluator = evaluator

        self.incumbent_cost = float(np.infty)

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
        path = os.path.join(base_path, benchmarklib)
        if benchmarklib == "yahpogym":
            path = os.path.join(path, "cluster", scenario, f"{dataset}.csv")
        elif benchmarklib == "mfpbench":
            path = os.path.join(path, "cluster", f"{scenario}.csv")
        return path

    def get_prior_data(
        self,
    ) -> pd.DataFrame:
        return pd.read_csv(self.prior_data_path)

    def on_ask_start(self, smbo: SMBO):
        "We add prior information, before the next iteration is started."
        smbo.intensifier.config_selector._acquisition_function.current_config_nuber = smbo.runhistory.finished

        if self.intervene(smbo):
            prior_configspace, origin_configpsace, cost, logging_config, superior_configuraiton = self.construct_prior(smbo)

            self.sample_from_prior(smbo, prior_configspace)

            prior_accepted, prior_mean_acq_value, origin_mean_acq_value = self.accept_prior(smbo, prior_configspace, origin_configpsace)

            if prior_accepted:
                self.set_prior(smbo, prior_configspace)
            self.log_prior(
                smbo=smbo,
                cost=cost,
                config=logging_config,
                prior_accepted=prior_accepted,
                prior_mean_acq_value=prior_mean_acq_value,
                origin_mean_acq_value=origin_mean_acq_value,
                superior_configuration=superior_configuraiton,
            )

        return super().on_ask_start(smbo)

    def sample_from_prior(self, smbo, prior_configspace):
        if self.n_prior_based_samples is not None:
            prior_samples = (
                prior_configspace.sample_configuration(size=self.n_prior_based_samples)
                if self.n_prior_based_samples != 1
                else [prior_configspace.sample_configuration(size=self.n_prior_based_samples)]
            )
            runner = smbo._runner
            target_function = runner._target_function

            for config in prior_samples:
                performance, runtime = target_function(config)
                trial_info = TrialInfo(config=config, instance=None, seed=0)
                trial_value = TrialValue(cost=performance, time=runtime)
                smbo.tell(trial_info, trial_value)

            smbo.intensifier.config_selector._acquisition_function.current_config_nuber = smbo.runhistory.finished

    def accept_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace, origin_configspace: ConfigurationSpace) -> bool:
        if self.validate_prior:
            if self.prior_validation_method == PriorValidationMethod.BASELINE_PERFECT.value:
                return self._accept_prior_baseline_perfect(), None, None

            current_incumbent = smbo.intensifier.get_incumbent()
            incumbent_configuration_dict = dict(current_incumbent)
            incumbent_configuration_space = build_prior_configuration_space(origin_configspace, incumbent_configuration_dict, prior_std_denominator=self.prior_std_denominator * self.prior_number)

            prior_samples = prior_configspace.sample_configuration(size=self.n_prior_validation_samples)
            incumbent_samples = incumbent_configuration_space.sample_configuration(size=self.n_prior_validation_samples)

            self.lcb.update(model=smbo.intensifier.config_selector._acquisition_function.model, num_data=smbo.runhistory.finished)
            lcb_prior_values = self.lcb(prior_samples).squeeze()
            lcb_incumbent_values = self.lcb(incumbent_samples).squeeze()
            if self.prior_validation_method == PriorValidationMethod.MANN_WHITNEY_U.value:
                raise ValueError("Mann-Whitney U test is depreceated.")
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
    def _accept_prior_baseline_perfect(self, smbo: SMBO, prior_configspace: ConfigurationSpace, origin_configspace: ConfigurationSpace) -> bool:
        pass

    @abstractmethod
    def intervene(self, smbo: SMBO) -> bool:
        pass

    def construct_prior(self, smbo: SMBO) -> Tuple[ConfigurationSpace, ConfigurationSpace, float, Dict, bool]:
        """
        Sets a new prior on the acquisition function and configspace.
        """
        self.prior_number += 1

        current_incumbent_config = smbo.intensifier.get_incumbent()
        incumbent_cost = smbo.runhistory.get_cost(current_incumbent_config)

        sampled_config = self.sample_prior(smbo, current_incumbent_config, incumbent_cost)
        prior_cost = sampled_config[COST_INDICATOR_COLUMN].values[0]

        # Check if the sampled configuration is better than the incumbent
        superior_configuration = bool(prior_cost < incumbent_cost)

        if sampled_config is None:
            raise ValueError("No prior configuration could be sampled.")

        cost = sampled_config[COST_INDICATOR_COLUMN].values[0]
        hyperparameter_config = sampled_config[[col for col in sampled_config.columns if col.startswith("config_")]]
        hyperparameter_config.columns = [col.replace("config_", "") for col in hyperparameter_config.columns]
        hyperparameter_config = hyperparameter_config.iloc[0].to_dict()

        origin_configspace = smbo._configspace
        prior_configspace = build_prior_configuration_space(origin_configspace, hyperparameter_config, prior_std_denominator=self.prior_std_denominator * self.prior_number)

        return prior_configspace, origin_configspace, cost, hyperparameter_config, superior_configuration

    def set_prior(self, smbo: SMBO, prior_configspace: ConfigurationSpace):
        smbo.intensifier.config_selector._acquisition_maximizer.dynamic_init(prior_configspace)
        smbo.intensifier.config_selector._acquisition_function.dynamic_init(
            acquisition_function=smbo.intensifier.config_selector._acquisition_function._acquisition_function,
            prior_configspace=prior_configspace,
            decay_beta=self.prior_decay_enumerator / self.prior_decay_denominator,
            prior_start=smbo.runhistory.finished,
        )

    @abstractmethod
    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> pd.DataFrame:
        """
        Samples a prior from the prior data.
        """

    def _sample_cluster(self, smbo, min_cluster, max_cluster, decay):
        vals = np.arange(min_cluster, max_cluster)
        probs = np.exp(-decay * (vals - min_cluster))
        probs /= probs.sum()
        return smbo.intensifier._rng.choice(vals, p=probs[::-1])

    def log_prior(self, smbo: SMBO, cost: float, config: Dict, prior_accepted: bool, prior_mean_acq_value: float, origin_mean_acq_value: float, superior_configuration: bool):
        """
        Logs the prior data.
        """
        if self.result_processor is not None:
            self.result_processor.process_logs(
                {
                    "priors": {
                        "prior_accepted": prior_accepted,
                        "superior_configuration": superior_configuration,
                        "cost": cost,
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

        sample = df.sample(random_state=rng)
        return sample


class DynaBOAbstractPriorCallback(AbstractPriorCallback):
    def __init__(
        self,
        prior_static_position: bool,
        prior_every_n_trials: int,
        prior_chance_theta: float,
        prior_at_start: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_trials_since_last_prior = 0
        self.prior_static_position = prior_static_position
        self.prior_every_n_trials = prior_every_n_trials
        self.prior_chance_theta = prior_chance_theta
        self.prior_at_start = prior_at_start

    def intervene(self, smbo: SMBO) -> bool:
        def _intervene_static_position():
            return smbo.runhistory.finished >= self.initial_design_size + 1 and (smbo.runhistory.finished - self.initial_design_size - 1) % self.prior_every_n_trials == 0  and self.prior_number < 4

        def _intervene_dynamic_position():
            if smbo.runhistory.finished < self.initial_design_size + 1:  # We do not intervene before the initial design is finished
                self.n_trials_since_last_prior += 1
                return False
            else:
                if smbo.runhistory.finished == self.initial_design_size + 1:  # We use the prior at the start of the optimization
                    if self.prior_at_start :
                        self.n_trials_since_last_prior = 0
                        return True
                else:
                    chance = 1 - np.exp(-self.prior_chance_theta * self.n_trials_since_last_prior)
                    if smbo.intensifier._rng.random() < chance:
                        self.n_trials_since_last_prior = 0
                        return True
                    else:
                        self.n_trials_since_last_prior += 1
                        return False

        if self.prior_static_position:
            return _intervene_static_position()
        else:
            return _intervene_dynamic_position()


class PiBOAbstractPriorCallback(AbstractPriorCallback):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

    def intervene(self, smbo):
        # To use the surrogate, we need to sample one additional config here
        return smbo.runhistory.finished == self.initial_design_size + 1

    def accept_prior(self, smbo, prior_configspace, origin_configspace):
        return True, None, None

    def _accept_prior_baseline_perfect(self) -> bool:
        return True


class WellPerformingPriorCallback(AbstractPriorCallback):
    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        # Locate current cost in the prior data column median_cost
        relevant_configs = self.prior_data[self.prior_data["median_cost"] <= incumbent_cost]

        min_cluster = relevant_configs["cluster"].min()
        max_cluster = relevant_configs["cluster"].max()

        if relevant_configs.empty:
            cluster = 0
        elif min_cluster == max_cluster:
            cluster = min_cluster
        else:
            cluster = self._sample_cluster(smbo, min_cluster, max_cluster, 0.1)

        # Select lowest cost configuration in the cluster
        relevant_configs = self.prior_data[self.prior_data["cluster"] == cluster].sort_values(COST_INDICATOR_COLUMN)[:1]
        return relevant_configs


class DynaBOWellPerformingPriorCallback(DynaBOAbstractPriorCallback, WellPerformingPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile

    def _accept_prior_baseline_perfect(self) -> bool:
        return True


class PiBOWellPerformingPriorCallback(PiBOAbstractPriorCallback, WellPerformingPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile


class MediumPerformingPriorCallback(AbstractPriorCallback):
    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        # Locate current cost in the prior data column median_cost
        relevant_configs = self.prior_data[self.prior_data["median_cost"] <= incumbent_cost]

        min_cluster = relevant_configs["cluster"].min()
        max_cluster = relevant_configs["cluster"].max()

        if relevant_configs.empty:
            cluster = 0
        elif min_cluster == max_cluster:
            cluster = min_cluster
        else:
            cluster = self._sample_cluster(smbo, min_cluster, max_cluster, 0.15)

        # Select lowest cost configuration in the cluster
        relevant_configs = self.prior_data[self.prior_data["cluster"] == cluster].sort_values(COST_INDICATOR_COLUMN)[:1]
        return self.draw_sample(relevant_configs, smbo.intensifier._rng)


class DynaBOMediumPriorCallback(DynaBOAbstractPriorCallback, MediumPerformingPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile
        self.helpful_prior = None

    def _accept_prior_baseline_perfect(self) -> bool:
        return self.helpful_prior

    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        prior = super().sample_prior(smbo, incumbent_config, incumbent_cost)
        prior_cost = prior[COST_INDICATOR_COLUMN].values[0]

        if prior_cost < incumbent_cost:
            self.helpful_prior = True
        else:
            self.helpful_prior = False

        return prior


class PiBOMediumPriorCallback(PiBOAbstractPriorCallback, MediumPerformingPriorCallback):
    def __init__(
        self,
        no_incumbent_percentile: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_incumbent_percentile = no_incumbent_percentile


class MisleadingPriorCallback(AbstractPriorCallback):
    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        prior_data = deepcopy(self.prior_data)

        closest_clusters = self.compute_closest_clusters(incumbent_config, prior_data)
        relevant_configs = prior_data[prior_data["cluster"].isin(closest_clusters["cluster"].values)]

        # Select cluster with lowest median cost
        min_median_cost = relevant_configs["median_cost"].min()
        relevant_configs = prior_data[prior_data["median_cost"] == min_median_cost]

        return self.draw_sample(relevant_configs, smbo.intensifier._rng)

    def compute_closest_clusters(self, incumbent_config: Configuration, prior_data: pd.DataFrame):
        # Preprocess Data
        centroid_configurations = prior_data[[col for col in prior_data.columns if col.startswith("medoid_config") or col == "cluster"]].drop_duplicates()
        centroid_configurations_only_config = centroid_configurations[[col for col in centroid_configurations.columns if col.startswith("medoid_config")]]
        centroid_configurations_only_config.columns = [col.replace("medoid_config_", "") for col in centroid_configurations_only_config.columns]

        # Create Distance Matrix
        keys = list(incumbent_config.config_space.keys())
        values = list()
        for key in keys:
            if key in incumbent_config.keys():
                values.append(incumbent_config[key])
            else:
                values.append(np.nan)
        gower_matrix_dataframe = pd.DataFrame(columns=list(keys), data=[values])
        gower_matrix_dataframe = pd.concat([gower_matrix_dataframe, centroid_configurations_only_config], ignore_index=True, axis=0)
        cat_features = [not pd.api.types.is_numeric_dtype(gower_matrix_dataframe[col]) for col in gower_matrix_dataframe.columns]
        distance_matrix = gower.gower_matrix(gower_matrix_dataframe.fillna(-1), cat_features=cat_features)

        # Select top 11 entries. The 10 closest clusters and the incumbent itself
        relevant_distances = distance_matrix[0, :]

        # Remove the incumbent itself
        closest_k_cluster_indexes = np.argsort(relevant_distances)[1:6]
        closest_k_clusters_centroids = gower_matrix_dataframe.iloc[closest_k_cluster_indexes]

        whole_entries = pd.merge(
            closest_k_clusters_centroids,
            centroid_configurations,
            left_on=list(closest_k_clusters_centroids.columns),
            right_on=[f"medoid_config_{hyperparameter}" for hyperparameter in incumbent_config.config_space.keys()],
            how="left",
        )

        return whole_entries[["cluster"]]


class DynaBOMisleadingPriorCallback(DynaBOAbstractPriorCallback, MisleadingPriorCallback):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.helpful_prior = None

    def _accept_prior_baseline_perfect(self) -> bool:
        return self.helpful_prior

    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        prior = super().sample_prior(smbo, incumbent_config, incumbent_cost)
        prior_cost = prior[COST_INDICATOR_COLUMN].values[0]

        if prior_cost < incumbent_cost:
            self.helpful_prior = True
        else:
            self.helpful_prior = False

        return prior


class PiBOMisleadingPriorCallback(PiBOAbstractPriorCallback, MisleadingPriorCallback):
    pass


class DeceivingPriorCallback(AbstractPriorCallback):
    relevant_cluster_lower_bound = 0.95
    relevant_cluster_upper_bound = 1

    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> Optional[pd.DataFrame]:
        # Select only the prior data in clusters between relevant_cluster_lower_bound and relevant_cluster_upper_bound
        relevant_clusters = self.prior_data[
            self.prior_data["cluster"].between(self.relevant_cluster_lower_bound * self.prior_data["cluster"].max(), self.relevant_cluster_upper_bound * self.prior_data["cluster"].max())
        ]["cluster"].unique()
        cluster = smbo.intensifier._rng.choice(relevant_clusters)
        # Select lowest cost configuration in the cluster
        relevant_configs = self.prior_data[self.prior_data["cluster"] == cluster].sort_values(COST_INDICATOR_COLUMN)[-1:]
        return relevant_configs


class DynaBODeceivingPriorCallback(DynaBOAbstractPriorCallback, DeceivingPriorCallback):
    def _accept_prior_baseline_perfect(self) -> bool:
        return False


class PiBODeceivingPriorCallback(PiBOAbstractPriorCallback, DeceivingPriorCallback):
    pass


class PriorOutOfRangeError(Exception):
    pass


class PiBOTestAllPriors(PiBOAbstractPriorCallback):
    def __init__(self, prior_number: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_number = prior_number
        self.prior_data = self.prior_data.sort_values(COST_INDICATOR_COLUMN)

    def sample_prior(self, smbo: SMBO, incumbent_config: Configuration, incumbent_cost: float) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.prior_data.iloc[self.prior_number].to_dict(), index=[0])
        except IndexError:
            raise PriorOutOfRangeError(f"Prior number {self.prior_number} is out of range. There are {self.prior_data.shape[0]} priors.")

    def _accept_prior_baseline_perfect(self) -> bool:
        return True
