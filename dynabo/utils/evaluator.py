import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from carps.utils.running import make_task
from ConfigSpace import Configuration, ConfigurationSpace
from omegaconf import OmegaConf
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from smac.runhistory import TrialInfo, TrialValue
from yahpo_gym import benchmark_set, local_config


class AbstractEvaluator:
    def __init__(
        self,
        scenario: str | int,
        dataset: int,
    ):
        self.scenario = scenario
        self.dataset = dataset

        self.accumulated_runtime = 0
        self.reasoning_runtime = 0

        self.incumbent_cost = None
        self.eval_counter = 0
        self.timeout_counter = 0

    def train(self, config: Configuration, seed: int = 0):
        performance, runtime = self._train(config=config, seed=seed)

        self.accumulated_runtime = round(self.accumulated_runtime + runtime, 3)

        if self.incumbent_cost is None or performance < self.incumbent_cost:
            self.incumbent_cost = performance

        return float(performance), float(runtime)

    @abstractmethod
    def _train(self, config: Configuration, seed: int = 0):
        pass

    @abstractmethod
    def get_configuration_space(self) -> ConfigurationSpace:
        pass  # TODO update this to get the correct data

    def get_metadata(self):
        return {
            "final_cost": self.incumbent_cost,
            "virtual_runtime": round(self.accumulated_runtime + self.reasoning_runtime, 3),
            "reasoning_runtime": round(self.reasoning_runtime, 3),
            "n_evaluations_computed": self.eval_counter,
        }

    @staticmethod
    @abstractmethod
    def get_fixed_hyperparameter_combinations(
        with_all_datasets: bool = True,
        medium_and_hard: bool = False,
        pibo: bool = False,
        dynabo: bool = False,
        baseline: bool = False,
        random: bool = False,
    ) -> List[Dict[str, Any]]:
        pass

    @staticmethod
    def extract_validate_prior_dict(
        validate_prior: bool,
        prior_validation_manwhitney: Optional[bool] = None,
        prior_validation_difference: Optional[bool] = None,
        n_prior_validation_samples: Optional[int] = None,
        prior_validation_manwhitney_p: Optional[float] = None,
        prior_validation_difference_threshold: Optional[float] = None,
    ):
        configs = list()
        if validate_prior:
            if prior_validation_manwhitney:
                configs += [
                    {
                        "validate_prior": True,
                        "prior_validation_method": "mann_whitney_u",
                        "n_prior_validation_samples": n_prior_validation_samples,
                        "prior_validation_manwhitney_p": prior_validation_manwhitney_p,
                        "prior_validation_difference_threshold": None,
                    }
                ]
            if prior_validation_difference:
                configs += [
                    {
                        "validate_prior": True,
                        "prior_validation_method": "difference",
                        "n_prior_validation_samples": n_prior_validation_samples,
                        "prior_validation_manwhitney_p": None,
                        "prior_validation_difference_threshold": x,
                    }
                    for x in prior_validation_difference_threshold
                ]
        else:
            configs += [
                {
                    "validate_prior": False,
                    "prior_validation_method": None,
                    "n_prior_validation_samples": None,
                    "prior_validation_manwhitney_p": None,
                    "prior_validation_difference_threshold": None,
                }
            ]
        return configs


YAHPOGYM_SCENARIO_OPTIONS = [
    "rbv2_ranger",
    "rbv2_xgboost",
    "rbv2_svm",
    "rbv2_glmnet",
    "lcbench",
    "nb301",
    "rbv2_aknn",
    "rbv2_rpart",
    "rbv2_super",
]


class YAHPOGymEvaluator(AbstractEvaluator):
    def __init__(self, scenario: str, dataset: int, metric="acc", runtime_metric_name="timetrain"):
        super().__init__(scenario=scenario, dataset=dataset)
        self.metric = metric
        self.runtime_metric_name = runtime_metric_name

        local_config._config = "benchmark_data/yahpo_data"
        self.benchmark = benchmark_set.BenchmarkSet(scenario=scenario, multithread=False)
        self.benchmark.set_instance(value=self.dataset)
        self.default_fidelity_config = self.benchmark.get_fidelity_space().get_default_configuration()

    def _train(self, config: Configuration, seed: int = 0):
        final_config = dict(config)

        for fidelity_param, value in self.default_fidelity_config.items():
            final_config[fidelity_param] = value
        final_config["task_id"] = self.dataset

        res = self.benchmark.objective_function(configuration=final_config)
        performance = round((-1) * res[0][self.metric], 6)

        runtime = round(res[0][self.runtime_metric_name], 3)
        return performance, runtime

    def get_configuration_space(self) -> ConfigurationSpace:
        return self.benchmark.get_opt_space(drop_fidelity_params=True)

    @staticmethod
    def get_datasets(with_all_datasets: bool, medium_and_hard: bool) -> List[str]:
        if not with_all_datasets ^ medium_and_hard:
            raise ValueError("Either with_all_datasets or medium_and_hard must be True, but not both.")

        scenario_dataset_combinations = []
        for scenario in YAHPOGYM_SCENARIO_OPTIONS:
            benchmark = benchmark_set.BenchmarkSet(scenario=scenario)
            if "val_accuracy" in benchmark.config.y_names:
                metric = "val_accuracy"
            elif "acc" in benchmark.config.y_names:
                metric = "acc"
            else:
                raise ValueError(f"Unknown metric in benchmark {benchmark.name}. Please check the benchmark configuration.")

            if with_all_datasets:
                datasets = benchmark.instances
            elif medium_and_hard:
                datasets = YAHPOGymEvaluator.get_medium_and_hard_datasets(scenario)

            scenario_dataset_combinations += [{"benchmarklib": "yahpogym", "scenario": scenario, "dataset": dataset, "metric": metric} for dataset in datasets]
        return scenario_dataset_combinations

    @staticmethod
    def get_medium_and_hard_datasets(scenario: str) -> List["str"]:
        difficulty_groups = pd.read_csv("plotting_data/yahpogym/difficulty_groups_one_seed.csv")
        medium_and_hard_df = difficulty_groups[(difficulty_groups["scenario"] == scenario) & (difficulty_groups["hard"] | difficulty_groups["medium"])]
        return medium_and_hard_df["dataset"].unique().tolist()


MFPBENCH_SCENARIO_OPTIONS = ["cifar100_wideresnet_2048", "imagenet_resnet_512", "lm1b_transformer_2048", "translatewmt_xformer_64"]


class MFPBenchEvaluator(AbstractEvaluator):
    def __init__(
        self,
        scenario,
        seed: int = 42,
    ):
        super().__init__(scenario=scenario, dataset=None)
        self.scenario = scenario

        # setup mfpbench config
        exp_config = OmegaConf.load("CARP-S/carps/configs/task/MFPBench/SO/pd1/" + self.scenario + ".yaml")
        exp_config.seed = seed
        self.task = make_task(exp_config)

        self.config_space = self.get_configuration_space()

    def _train(self, config: Configuration, seed: int = 0):
        # We use the full fidelity space
        ti = TrialInfo(config=config, budget=1.0, seed=seed)
        res = self.task.objective_function.evaluate(ti)

        performance = round(float(res.cost), 6)
        runtime = round(float(res.virtual_time), 3)

        return float(performance), float(runtime)

    def get_configuration_space(self) -> ConfigurationSpace:
        return self.task.objective_function.configspace

    @staticmethod
    def get_datasets(with_all_datasets: bool, medium_and_hard: bool) -> List[Dict[str, Any]]:
        return [{"benchmarklib": "mfpbench", "scenario": scenario, "dataset": None, "metric": "cost"} for scenario in MFPBENCH_SCENARIO_OPTIONS]


def fill_table(
    py_experimenter: PyExperimenter,
    common_parameters: Dict[str, List[Any]],
    benchmarklib: str,
    benchmark_parameters: Dict[str, Any],
    approach: str,
    approach_parameters: Optional[Dict[str, Any]],
):
    common_dict = extract_common_config(
        acquisition_function=common_parameters["acquisition_function"],
        timeout_total=common_parameters["timeout_total"],
        n_trials=common_parameters["n_trials"],
        initial_design__n_configs_per_hyperparameter=common_parameters["initial_design__n_configs_per_hyperparameter"],
        initial_design__max_ratio=common_parameters["initial_design__max_ratio"],
        seeds=common_parameters["seed"],
    )

    benchmark_dict = extract_benchmark_config(
        benchmarklib=benchmarklib,
        with_all_datasets=benchmark_parameters["with_all_datasets"],
        medium_and_hard=benchmark_parameters["medium_and_hard"],
    )

    if approach == "baseline":
        approach_dict = get_baseline_dict()
    elif approach == "random":
        approach_dict = get_random_dict()
    elif approach == "pibo":
        approach_dict = get_pibo_dict(
            prior_kind_choices=approach_parameters["prior_kind_choices"],
            no_incumbent_percentile=approach_parameters["no_incumbent_percentile"],
            prior_std_denominator=approach_parameters["prior_std_denominator"],
            prior_decay_enumerator_choices=approach_parameters["prior_decay_enumerator_choices"],
            prior_decay_denominator=approach_parameters["prior_decay_denominator"],
        )
    elif approach == "dynabo":
        approach_dict = get_dynabo_dict(
            # Prior configuration
            prior_kind_choices=approach_parameters["prior_kind_choices"],
            no_incumbent_percentile=approach_parameters["no_incumbent_percentile"],
            prior_std_denominator=approach_parameters["prior_std_denominator"],
            # Prior location
            prior_static_position=approach_parameters["prior_static_position"],
            prior_every_n_trials_choices=approach_parameters["prior_every_n_trials_choices"],
            prior_at_start_choices=approach_parameters["prior_at_start_choices"],
            prior_chance_theta_choices=approach_parameters["prior_chance_theta_choices"],
            # Decay parameters
            prior_decay_enumerator_choices=approach_parameters["prior_decay_enumerator_choices"],
            prior_decay_denominator=approach_parameters["prior_decay_denominator"],
            # Validation parameters
            validate_prior_choices=approach_parameters["validate_prior_choices"],
            n_prior_validation_samples=approach_parameters["n_prior_validation_samples"],
            n_prior_based_samples=approach_parameters["n_prior_based_samples"],
            prior_validation_method_choices=approach_parameters["prior_validation_method_choices"],
            prior_validation_manwhitney_p_choices=approach_parameters["prior_validation_manwhitney_p_choices"],
            prior_validation_difference_threshold_choices=approach_parameters["prior_validation_difference_threshold_choices"],
        )

    # Join all combinations of benchmark_dict and approach_dict
    fixed_parameter_combinations = []
    for benchmark_combination in benchmark_dict:
        for approach_combination in approach_dict:
            fixed_parameter_combinations.append({**benchmark_combination, **approach_combination})

    py_experimenter.fill_table_from_combination(parameters=common_dict, fixed_parameter_combinations=fixed_parameter_combinations)


def fill_table_ablate_priors(
    py_experimenter: PyExperimenter,
    common_parameters: Dict[str, List[Any]],
    benchmarklib: str,
    benchmark_parameters: Dict[str, Any],
    approach: str,
    approach_parameters: Optional[Dict[str, Any]],
    prior_number: int,
):
    common_dict = extract_common_config(
        acquisition_function=common_parameters["acquisition_function"],
        timeout_total=common_parameters["timeout_total"],
        n_trials=common_parameters["n_trials"],
        initial_design__n_configs_per_hyperparameter=common_parameters["initial_design__n_configs_per_hyperparameter"],
        initial_design__max_ratio=common_parameters["initial_design__max_ratio"],
        seeds=common_parameters["seed"],
    )

    benchmark_dict = extract_benchmark_config(
        benchmarklib=benchmarklib,
        with_all_datasets=benchmark_parameters["with_all_datasets"],
        medium_and_hard=benchmark_parameters["medium_and_hard"],
    )

    if approach == "baseline":
        approach_dict = get_baseline_dict()
    elif approach == "random":
        approach_dict = get_random_dict()
    elif approach == "pibo":
        approach_dict = get_pibo_dict(
            prior_kind_choices=approach_parameters["prior_kind_choices"],
            no_incumbent_percentile=approach_parameters["no_incumbent_percentile"],
            prior_std_denominator=approach_parameters["prior_std_denominator"],
            prior_decay_enumerator_choices=approach_parameters["prior_decay_enumerator_choices"],
            prior_decay_denominator=approach_parameters["prior_decay_denominator"],
        )
        for entry in approach_dict:
            entry["prior_number"] = prior_number
    elif approach == "dynabo":
        approach_dict = get_dynabo_dict(
            # Prior configuration
            prior_kind_choices=approach_parameters["prior_kind_choices"],
            no_incumbent_percentile=approach_parameters["no_incumbent_percentile"],
            prior_std_denominator=approach_parameters["prior_std_denominator"],
            # Prior location
            prior_static_position=approach_parameters["prior_static_position"],
            prior_every_n_trials_choices=approach_parameters["prior_every_n_trials_choices"],
            prior_at_start_choices=approach_parameters["prior_at_start_choices"],
            prior_chance_theta_choices=approach_parameters["prior_chance_theta_choices"],
            # Decay parameters
            prior_decay_enumerator_choices=approach_parameters["prior_decay_enumerator_choices"],
            prior_decay_denominator=approach_parameters["prior_decay_denominator"],
            # Validation parameters
            validate_prior_choices=approach_parameters["validate_prior_choices"],
            n_prior_validation_samples=approach_parameters["n_prior_validation_samples"],
            prior_validation_method_choices=approach_parameters["prior_validation_method_choices"],
            prior_validation_manwhitney_p_choices=approach_parameters["prior_validation_manwhitney_p_choices"],
            prior_validation_difference_threshold_choices=approach_parameters["prior_validation_difference_threshold_choices"],
        )

    # Join all combinations of benchmark_dict and approach_dict
    fixed_parameter_combinations = []
    for benchmark_combination in benchmark_dict:
        for approach_combination in approach_dict:
            fixed_parameter_combinations.append({**benchmark_combination, **approach_combination})

    py_experimenter.fill_table_from_combination(parameters=common_dict, fixed_parameter_combinations=fixed_parameter_combinations)


def extract_common_config(
    acquisition_function: List[str],
    timeout_total: List[int],
    n_trials: List[int],
    initial_design__n_configs_per_hyperparameter: List[int],
    initial_design__max_ratio: List[float],
    seeds: List[int],
) -> Dict[str, List[Any]]:
    return {
        "acquisition_function": acquisition_function,
        "timeout_total": timeout_total,
        "n_trials": n_trials,
        "initial_design__n_configs_per_hyperparameter": initial_design__n_configs_per_hyperparameter,
        "initial_design__max_ratio": initial_design__max_ratio,
        "seed": seeds,
    }


def extract_benchmark_config(
    benchmarklib: str,
    with_all_datasets: bool,
    medium_and_hard: bool,
) -> List[Dict[str, Any]]:
    if benchmarklib not in ["yahpogym", "mfpbench"]:
        raise ValueError(f"Benchmarklib {benchmarklib} is not supported. Supported benchmarklibs are: ['yahpogym', 'mfpbench']")

    if benchmarklib == "yahpogym":
        return YAHPOGymEvaluator.get_datasets(
            with_all_datasets=with_all_datasets,
            medium_and_hard=medium_and_hard,
        )

    elif benchmarklib == "mfpbench":
        return MFPBenchEvaluator.get_datasets(
            with_all_datasets=with_all_datasets,
            medium_and_hard=medium_and_hard,
        )

    else:
        raise NotImplementedError(f"Benchmarklib {benchmarklib} is not implemented.")


def get_baseline_dict() -> List[Dict[str, bool]]:
    return [{"baseline": True, "random": False, "pibo": False, "dynabo": False}]


def get_random_dict() -> List[Dict[str, bool]]:
    return [{"baseline": False, "random": True, "pibo": False, "dynabo": False}]


def get_pibo_dict(
    prior_kind_choices: List[str],
    no_incumbent_percentile: float,
    prior_std_denominator: int,
    prior_decay_enumerator_choices: List[int],
    prior_decay_denominator: int,
):
    return [
        {
            "pibo": True,
            "dynabo": False,
            "baseline": False,
            "random": False,
            "prior_decay_enumerator": prior_decay_enumerator,
            "prior_decay_denominator": prior_decay_denominator,
            "prior_kind": prior_kind,
            "no_incumbent_percentile": no_incumbent_percentile,
            "prior_std_denominator": prior_std_denominator,
            "prior_static_position": None,
            "prior_every_n_trials": None,
            "prior_at_start": None,
            "prior_chance_theta": None,
            "validate_prior": None,
            "prior_validation_method": None,
            "n_prior_validation_samples": None,
            "n_prior_based_samples": None,
            "prior_validation_manwhitney_p": None,
            "prior_validation_difference_threshold": None,
        }
        for prior_kind in prior_kind_choices
        for prior_decay_enumerator in prior_decay_enumerator_choices
    ]


def get_dynabo_dict(
    prior_kind_choices: List[str],
    no_incumbent_percentile: float,
    prior_std_denominator: int,
    prior_static_position: bool,
    prior_every_n_trials_choices: List[int],
    prior_chance_theta_choices: List[float],
    prior_decay_enumerator_choices: List[int],
    prior_decay_denominator: int,
    prior_at_start_choices: List[bool],
    validate_prior_choices: List[bool],
    prior_validation_method_choices: List[str],
    n_prior_validation_samples: List[int],
    n_prior_based_samples: List[int],
    prior_validation_manwhitney_p_choices: List[float],
    prior_validation_difference_threshold_choices: List[float],
) -> List[Dict[str, Any]]:
    """Generate a list of DynaBO configurations with different validation settings.

    Args:
        prior_decay_enumerator: Numerator for prior decay calculation
        prior_decay_denominator: Denominator for prior decay calculation
        prior_every_n_trials: Frequency of prior updates
        prior_kind_choices: List of prior types to use
        validate_prior_choices: Whether to validate priors
        prior_validation_method_choices: Validation methods to use
        n_prior_validation_samples: Number of samples for validation
        n_prior_based_samples: Number of trials samples for prior validation
        prior_validation_manwhitney_p_choices: P-values for Mann-Whitney test
        prior_validation_difference_threshold_choices: Thresholds for difference test

    Returns:
        List[Dict[str, Any]]: List of configuration dictionaries
    """
    dynabo_configs: List[Dict[str, Any]] = []

    def create_base_config(
        prior_decay_enumerator: int,
        prior_decay_denominator: int,
        prior_kind: str,
        no_incumbent_percentile: float,
        prior_std_denominator: int,
        validate_prior: bool,
    ) -> Dict[str, Any]:
        """Create a base configuration dictionary."""
        return {
            "baseline": False,
            "random": False,
            "pibo": False,
            "dynabo": True,
            "prior_std_denominator": prior_std_denominator,
            "prior_decay_enumerator": prior_decay_enumerator,
            "prior_decay_denominator": prior_decay_denominator,
            "prior_kind": prior_kind,
            "no_incumbent_percentile": no_incumbent_percentile,
            "validate_prior": validate_prior,
        }

    def add_position_configs(
        base_config: Dict[str, Any],
        prior_static_position: bool,
        prior_every_n_trials_choices: List[int],
        prior_at_start_choices: List[bool],
        prior_chance_theta_choices: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Add position-related configurations to a base config.

        Args:
            base_config: Base configuration dictionary
            prior_static_position: Whether to use static position
            prior_every_n_trials: Number of trials between prior updates for static position
            prior_at_start_choices: List of choices for prior_at_start when not using static position
            prior_chance_theta_choices: List of choices for prior_chance_theta when not using static position

        Returns:
            List of configurations with different position settings
        """
        configs = []

        if prior_static_position:  # If the start and end position are static, we only need to add one config
            for prior_every_n_trials in prior_every_n_trials_choices:
                base_copy = deepcopy(base_config)
                base_copy["prior_static_position"] = True
                base_copy["prior_every_n_trials"] = prior_every_n_trials
                base_copy["prior_at_start"] = None
                base_copy["prior_chance_theta"] = None
                configs.append(base_copy)
        else:  # If the start and end position are not static, we need to add all combinations of prior_at_start and prior_chance_theta
            for prior_at_start in prior_at_start_choices:
                for prior_chance_theta in prior_chance_theta_choices:
                    base_copy = deepcopy(base_config)
                    base_copy["prior_static_position"] = False
                    base_copy["prior_every_n_trials"] = None
                    base_copy["prior_at_start"] = prior_at_start
                    base_copy["prior_chance_theta"] = prior_chance_theta
                    configs.append(base_copy)

        return configs

    def add_validation_method(
        preliminary_configs: List[Dict[str, Any]],
        validate_prior: bool,
        prior_validation_method_choices: List[str],
        n_prior_validation_samples: int,
    ) -> List[Dict[str, Any]]:
        """
        Add validation method configurations to a list of preliminary configs.

        Args:
            preliminary_configs: List of preliminary configuration dictionaries
            validate_prior: Whether to validate the prior
            prior_validation_method_choices: List of validation methods to consider
            n_prior_validation_samples: Number of samples for validation

        Returns:
            List of configurations with different validation methods
        """
        final_configs = []

        for base_config in preliminary_configs:  # For all considered configurations, we need to add all combinations of validation methods and sample sizes
            if not validate_prior:
                config = deepcopy(base_config)
                config["prior_validation_method"] = None
                config["n_prior_validation_samples"] = None
                config["n_prior_based_samples"] = None
                config["prior_validation_manwhitney_p"] = None
                config["prior_validation_difference_threshold"] = None
                final_configs.append(config)
                continue

            for validation_method in prior_validation_method_choices:
                if validation_method == "mann_whitney_u":
                    for config in _add_mann_whitney_configs(deepcopy(base_config), n_prior_validation_samples=n_prior_validation_samples, n_prior_based_samples=n_prior_based_samples):
                        final_configs.append(config)
                elif validation_method == "difference":
                    for config in _add_difference_configs(deepcopy(base_config), n_prior_validation_samples=n_prior_validation_samples, n_prior_based_samples=n_prior_based_samples):
                        final_configs.append(config)
                elif validation_method == "baseline_perfect":
                    for config in _add_baseline_perfect_configs(deepcopy(base_config), n_prior_based_samples=n_prior_based_samples):
                        final_configs.append(config)

        return final_configs

    def _add_mann_whitney_configs(base_config: Dict[str, Any], n_prior_validation_samples: int, n_prior_based_samples: int) -> Generator[Dict[str, Any], None, None]:
        """Add configurations for Mann-Whitney validation method."""
        base_config["prior_validation_method"] = "mann_whitney_u"
        base_config["n_prior_validation_samples"] = n_prior_validation_samples
        base_config["n_prior_based_samples"] = n_prior_based_samples
        for p_value in prior_validation_manwhitney_p_choices:
            config = deepcopy(base_config)
            config["prior_validation_manwhitney_p"] = p_value
            config["prior_validation_difference_threshold"] = None
            yield config

    def _add_difference_configs(base_config: Dict[str, Any], n_prior_validation_samples: int, n_prior_based_samples: int) -> Generator[Dict[str, Any], None, None]:
        """Add configurations for difference validation method."""
        base_config["prior_validation_method"] = "difference"
        base_config["n_prior_validation_samples"] = n_prior_validation_samples
        base_config["n_prior_based_samples"] = n_prior_based_samples
        for threshold in prior_validation_difference_threshold_choices:
            config = deepcopy(base_config)
            config["prior_validation_manwhitney_p"] = None
            config["prior_validation_difference_threshold"] = threshold
            yield config

    def _add_baseline_perfect_configs(base_config: Dict[str, Any], n_prior_based_samples: int) -> Generator[Dict[str, Any], None, None]:
        """Add configurations for baseline perfect validation method."""
        config = deepcopy(base_config)
        config["prior_validation_method"] = "baseline_perfect"
        config["n_prior_validation_samples"] = None
        config["n_prior_based_samples"] = n_prior_based_samples
        config["prior_validation_manwhitney_p"] = None
        config["prior_validation_difference_threshold"] = None

        yield config

    # Generate configurations
    for prior_kind in prior_kind_choices:
        for prior_decay_enumerator in prior_decay_enumerator_choices:
            for validate_prior in validate_prior_choices:
                base_config = create_base_config(
                    prior_decay_enumerator=prior_decay_enumerator,
                    prior_decay_denominator=prior_decay_denominator,
                    prior_kind=prior_kind,
                    no_incumbent_percentile=no_incumbent_percentile,
                    prior_std_denominator=prior_std_denominator,
                    validate_prior=validate_prior,
                )

                preliminary_configs = add_position_configs(base_config, prior_static_position, prior_every_n_trials_choices, prior_at_start_choices, prior_chance_theta_choices)

                validation_configs = add_validation_method(preliminary_configs, validate_prior, prior_validation_method_choices, n_prior_validation_samples)
                dynabo_configs.extend(validation_configs)

    return dynabo_configs


def ask_tell_opt(smac: HyperparameterOptimizationFacade, evaluator: AbstractEvaluator, result_processor: ResultProcessor, timeout: int):
    while smac.runhistory.finished < smac.scenario.n_trials:
        start_ask = time.time()
        trial_info: TrialInfo = smac.ask()
        end_ask = time.time()

        # add runtime for ask
        ask_runtime = round(end_ask - start_ask, 3)
        evaluator.reasoning_runtime += ask_runtime

        cost, runtime = evaluator.train(trial_info.config)
        trial_value = TrialValue(cost=cost, time=runtime)

        start_tell = time.time()
        smac.tell(info=trial_info, value=trial_value)
        end_tell = time.time()

        # add runtime for tell
        tell_runtime = round(end_tell - start_tell, 3)
        evaluator.reasoning_runtime += tell_runtime
