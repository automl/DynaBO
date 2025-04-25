import time
from abc import abstractmethod
from typing import List, Optional

import ioh
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace, Float
from py_experimenter.result_processor import ResultProcessor
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from smac.runhistory import TrialInfo, TrialValue
from yahpo_gym import benchmark_set


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
            "final_performance": -1 * self.incumbent_cost,
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
    ):
        pass


class YAHPOGymEvaluator(AbstractEvaluator):
    def __init__(self, scenario: str, dataset: int, metric="acc", runtime_metric_name="timetrain", inverted_cost: bool = False):
        super().__init__(scenario=scenario, dataset=dataset)
        self.metric = metric
        self.runtime_metric_name = runtime_metric_name
        self.inverted_cost = inverted_cost

        self.benchmark = benchmark_set.BenchmarkSet(scenario=scenario)
        self.benchmark.set_instance(value=self.dataset)
        self.default_fidelity_config = self.benchmark.get_fidelity_space().get_default_configuration()

    def _train(self, config: Configuration, seed: int = 0):
        final_config = dict(config)

        for fidelity_param, value in self.default_fidelity_config.items():
            final_config[fidelity_param] = value
        final_config["task_id"] = self.dataset

        res = self.benchmark.objective_function(configuration=final_config)
        performance = round((-1) * res[0][self.metric], 6)

        # If we utilize inverted cost, invert the performance values
        if self.inverted_cost:
            performance = -1 * performance

        runtime = round(res[0][self.runtime_metric_name], 3)
        return performance, runtime

    def get_configuration_space(self) -> ConfigurationSpace:
        return self.benchmark.get_opt_space(drop_fidelity_params=True)

    @staticmethod
    def get_fixed_hyperparameter_combinations(
        acquisition_function: str,
        with_all_datasets: bool,
        medium_and_hard: bool,
        pibo: bool,
        dynabo: bool,
        baseline: bool,
        random: bool,
        decay_enumerator: int,
        validate_prior: bool,
        prior_validation_manwhitney: Optional[bool],
        prior_validation_difference: Optional[bool],
        n_prior_validation_samples: Optional[int],
        prior_validation_manwhitney_p: Optional[float],
        prior_validation_difference_threshold: Optional[float],
    ):
        jobs = []

        # Add all YAHPO-Gym Evaluations
        for scenario in [
            "rbv2_ranger",
            "rbv2_xgboost",
            "rbv2_svm",
            "rbv2_glmnet",
            "lcbench",
            "nb301",
            "rbv2_aknn",
            "rbv2_rpart",
            "rbv2_super",
        ]:
            # asset pibo, baseliiine or dynabo is set
            assert pibo or dynabo or baseline or random
            job = []
            bench = benchmark_set.BenchmarkSet(scenario=scenario)
            if "val_accuracy" in bench.config.y_names:
                metric = "val_accuracy"
            elif "acc" in bench.config.y_names:
                metric = "acc"
            else:
                metric = "unknown"

            if baseline:
                job += [{"pibo": False, "dynabo": False, "baseline": True, "acquisition_function": acquisition_function, "random": False}]
            if pibo:
                configs = YAHPOGymEvaluator.extract_validate_prior_dict(validate_prior=False)
                job += [
                    {"pibo": True, "dynabo": False, "baseline": False, "acquisition_function": acquisition_function, "random": False, "prior_decay_enumerator": decay_enumerator, **config}
                    for config in configs
                ]
            if dynabo:
                configs = YAHPOGymEvaluator.extract_validate_prior_dict(
                    validate_prior=validate_prior,
                    prior_validation_manwhitney=prior_validation_manwhitney,
                    prior_validation_difference=prior_validation_difference,
                    n_prior_validation_samples=n_prior_validation_samples,
                    prior_validation_manwhitney_p=prior_validation_manwhitney_p,
                    prior_validation_difference_threshold=prior_validation_difference_threshold,
                )
                job += [
                    {"pibo": False, "dynabo": True, "baseline": False, "acquisition_function": acquisition_function, "random": False, "prior_decay_enumerator": decay_enumerator, **config}
                    for config in configs
                ]
            if random:
                job += [{"pibo": False, "dynabo": False, "baseline": False, "random": True}]

            if with_all_datasets:
                # create ablation and ds_tunability jobs
                new_job = [{"scenario": scenario, "dataset": dataset, "metric": metric} for dataset in bench.instances]
                # combine job with new_job
            elif medium_and_hard:
                medium_and_hard_datasets = get_medium_and_hard_datasets(scenario)
                new_job = [{"scenario": scenario, "dataset": dataset, "metric": metric} for dataset in medium_and_hard_datasets]
            else:
                new_job = [{"scenario": scenario, "dataset": "all", "metric": metric}]

            jobs += [dict(**j, **nj) for j in job for nj in new_job]

        return jobs

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
                        "prior_validation_difference_threshold": prior_validation_difference_threshold,
                    }
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


class BBOBEvaluator(AbstractEvaluator):
    def __init__(self, scenario: int, dataset: str, dimension: int):
        super().__init__(int(scenario), dataset)
        self.dimension = dimension

        self.problem = ioh.get_problem(
            fid=int(scenario),
            instance=dataset,
            dimension=dimension,
            # problem_type=ProblemType.BBOB,
        )

    def get_configuration_space(self):
        upper_bounds = self.problem.bounds.ub
        lower_bounds = self.problem.bounds.lb
        hps = [Float(name=f"x{i}", bounds=[lower_bounds[i], upper_bounds[i]]) for i in range(self.dimension)]
        configuration_space = ConfigurationSpace()
        configuration_space.add(hps)
        return configuration_space

    def _train(self, config: Configuration, seed: int):
        values = list(config.values())
        performance = self.problem(values)
        return performance, 0

    @staticmethod
    def get_fixed_hyperparameter_combinations(
        with_all_datasets: bool = True,
        medium_and_hard: bool = False,
        pibo: bool = False,
        dynabo: bool = False,
        baseline: bool = False,
        random: bool = False,
    ):
        pass


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


def get_yahpo_fixed_parameter_combinations(
    benchmarklib: str,
    acquisition_function: str,
    with_all_datasets: bool,
    medium_and_hard: bool,
    pibo: bool,
    dynabo: bool,
    baseline: bool,
    random: bool,
    decay_enumerator: Optional[int] = None,
    validate_prior: Optional[bool] = None,
    prior_validation_manwhitney: Optional[bool] = None,
    prior_validation_difference: Optional[bool] = None,
    n_prior_validation_samples: Optional[int] = None,
    prior_validation_manwhitney_p: Optional[float] = None,
    prior_validation_difference_threshold: Optional[float] = None,
):
    if benchmarklib == "yahpogym":
        jobs = YAHPOGymEvaluator.get_fixed_hyperparameter_combinations(
            acquisition_function=acquisition_function,
            with_all_datasets=with_all_datasets,
            medium_and_hard=medium_and_hard,
            pibo=pibo,
            dynabo=dynabo,
            baseline=baseline,
            random=random,
            decay_enumerator=decay_enumerator,
            validate_prior=validate_prior,
            prior_validation_manwhitney=prior_validation_manwhitney,
            prior_validation_difference=prior_validation_difference,
            n_prior_validation_samples=n_prior_validation_samples,
            prior_validation_manwhitney_p=prior_validation_manwhitney_p,
            prior_validation_difference_threshold=prior_validation_difference_threshold,
        )
    elif benchmarklib == "bbob":
        jobs = BBOBEvaluator.get_fixed_hyperparameter_combinations(
            with_all_datasets=with_all_datasets,
            medium_and_hard=medium_and_hard,
            pibo=pibo,
            dynabo=dynabo,
            baseline=baseline,
            random=random,
        )
    return jobs


def get_medium_and_hard_datasets(scenario: str) -> List["str"]:
    difficulty_groups = pd.read_csv("plotting_data/difficulty_groups_one_seed.csv")
    medium_and_hard_df = difficulty_groups[(difficulty_groups["scenario"] == scenario) & (difficulty_groups["hard"] | difficulty_groups["medium"])]
    return medium_and_hard_df["dataset"].unique().tolist()
