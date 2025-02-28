import time
from abc import abstractmethod
from typing import List

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from py_experimenter.result_processor import ResultProcessor
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from smac.runhistory import StatusType, TrialInfo, TrialValue
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

    @abstractmethod
    def train(self, configuration: Configuration, seed: int = 0):
        pass  # TODO update this to to contain the general function

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


class YAHPOGymEvaluator(AbstractEvaluator):
    def __init__(
        self,
        scenario,
        dataset,
        metric="acc",
        runtime_metric_name="timetrain",
    ):
        super().__init__(scenario=scenario, dataset=dataset)
        self.metric = metric
        self.runtime_metric_name = runtime_metric_name

        self.benchmark = benchmark_set.BenchmarkSet(scenario=scenario, check=False)
        self.benchmark.set_instance(value=self.dataset)

    def train(self, config: Configuration, seed: int = 0):
        self.eval_counter += 1
        config_dict = dict(config)

        def_conf = dict(self.benchmark.get_opt_space().get_default_configuration())
        for key, value in config_dict.items():
            def_conf[key] = value

        res = self.benchmark.objective_function(configuration=def_conf)
        performance = round((-1) * res[0][self.metric], 6)
        runtime = round(res[0][self.runtime_metric_name], 3)

        self.accumulated_runtime = round(self.accumulated_runtime + runtime, 3)

        if self.incumbent_cost is None or performance < self.incumbent_cost:
            self.incumbent_cost = performance

        return float(performance), float(runtime)

    def get_configuration_space(self) -> ConfigurationSpace:
        return self.benchmark.get_opt_space(drop_fidelity_params=True)


def ask_tell_opt(smac: HyperparameterOptimizationFacade, evaluator: YAHPOGymEvaluator, result_processor: ResultProcessor, timeout: int):
    while smac.runhistory.finished < smac.scenario.n_trials:
        start_ask = time.time()
        trial_info: TrialInfo = smac.ask()
        end_ask = time.time()

        # add runtime for ask
        ask_runtime = round(end_ask - start_ask, 3)
        evaluator.reasoning_runtime += ask_runtime

        try:
            cost, runtime = evaluator.train(dict(trial_info.config))
            trial_value = TrialValue(cost=cost, time=runtime)
        except Exception:
            trial_value = TrialValue(cost=0, status=StatusType.TIMEOUT)

        start_tell = time.time()
        smac.tell(info=trial_info, value=trial_value)
        end_tell = time.time()

        # add runtime for tell
        tell_runtime = round(end_tell - start_tell, 3)
        evaluator.reasoning_runtime += tell_runtime


def get_yahpo_fixed_parameter_combinations(
    with_all_datasets: bool = True,
    medium_and_hard: bool = False,
    pibo: bool = False,
    dynabo: bool = False,
    baseline: bool = False,
    random: bool = False,
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
        bench = benchmark_set.BenchmarkSet(scenario=scenario)

        if "val_accuracy" in bench.config.y_names:
            metric = "val_accuracy"
        elif "acc" in bench.config.y_names:
            metric = "acc"
        else:
            metric = "unknown"

        # asset pibo, baseliiine or dynabo is set
        assert pibo or dynabo or baseline or random
        job = []

        if baseline:
            job += [{"pibo": False, "dynabo": False, "baseline": True, "random": False}]
        if pibo:
            job += [{"pibo": True, "dynabo": False, "baseline": False, "random": False, "prior_decay_enumerator": 200}]
        if dynabo:
            job += [{"pibo": False, "dynabo": True, "baseline": False, "random": False, "prior_decay_enumerator": 50}]
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


def get_medium_and_hard_datasets(scenario: str) -> List["str"]:
    difficulty_groups = pd.read_csv("plotting_data/difficulty_groups_one_seed.csv")
    medium_and_hard_df = difficulty_groups[(difficulty_groups["scenario"] == scenario) & (difficulty_groups["hard"] | difficulty_groups["medium"])]
    return medium_and_hard_df["dataset"].tolist()
