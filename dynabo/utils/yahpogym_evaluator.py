import json
import time

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from py_experimenter.result_processor import ResultProcessor
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from smac.runhistory import StatusType, TrialInfo, TrialValue
from yahpo_gym import benchmark_set


class YAHPOGymEvaluator:
    def __init__(
        self,
        scenario,
        dataset,
        internal_timeout=-1,
        metric="acc",
        runtime_metric_name="timetrain",
        result_processor: ResultProcessor = None,
    ):
        self.scenario = scenario
        self.dataset = dataset
        self.metric = metric
        self.runtime_metric_name = runtime_metric_name
        self.internal_timeout = internal_timeout
        self.result_processor = result_processor

        self.benchmark = benchmark_set.BenchmarkSet(scenario=scenario, check=False)
        self.benchmark.set_instance(value=self.dataset)

        self.accumulated_runtime = 0
        self.reasoning_runtime = 0

        self.incumbent_trace = list()
        self.incumbent_cost = None
        self.eval_counter = 0
        self.timeout_counter = 0

    def train(self, config: Configuration, seed: int = 0):
        self.eval_counter += 1
        config_dict = dict(config)

        if self.eval_counter % 100 == 0 and self.result_processor is not None:
            from datetime import datetime

            now = datetime.now()
            self.result_processor.process_results({"num_evaluations": str(self.eval_counter) + " " + now.strftime("%m/%d/%Y, %H:%M:%S")})

        def_conf = dict(self.benchmark.get_opt_space().get_default_configuration())
        for key, value in config_dict.items():
            def_conf[key] = value

        res = self.benchmark.objective_function(configuration=def_conf)
        performance = round((-1) * res[0][self.metric], 6)
        runtime = round(res[0][self.runtime_metric_name], 3)

        # check whether internal evaluation timeout is set
        if self.internal_timeout != -1:
            # check whether timeout is hit
            if runtime > self.internal_timeout:
                self.accumulated_runtime += self.internal_timeout
                self.timeout_counter += 1
                raise Exception("Internal timeout exceeded")

        self.accumulated_runtime = round(self.accumulated_runtime + runtime, 3)

        if self.incumbent_cost is None or performance < self.incumbent_cost:
            self.incumbent_cost = performance
            incumbent_tuple = (
                round(self.reasoning_runtime + self.accumulated_runtime, 3),
                (-1) * performance,
                self.eval_counter,
                def_conf,
            )
            print("new incumbent:", incumbent_tuple)
            self.incumbent_trace.append(incumbent_tuple)
            if self.result_processor is not None:
                self.result_processor.process_results(
                    {
                        "incumbent_trace": json.dumps(self.incumbent_trace),
                    }
                )

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
    medium_and_hard_datasets: bool = False,
    pibo: bool = False,
    dynabo: bool = False,
    baseline: bool = False,
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
        assert pibo or dynabo or baseline
        job = []

        if baseline:
            job += [{"pibo": False, "dynabo": False, "prior_decay_enumerator": 50}]
        elif pibo:
            job += [{"pibo": True, "dynabo": False, "prior_decay_enumerator": 50}]
        elif dynabo:
            job += [{"pibo": False, "dynabo": True, "prior_decay_enumerator": 50}]

        if with_all_datasets:
            # create ablation and ds_tunability jobs
            new_job = [{"scenario": scenario, "dataset": dataset, "metric": metric} for dataset in bench.instances]
            # combine job with new_job
        elif medium_and_hard_datasets:
            medium_df = get_medium_and_hard_datasets(scenario)
            new_job = [{"scenario": scenario, "dataset": dataset, "metric": metric} for dataset in medium_df]
        else:
            new_job = [{"scenario": scenario, "dataset": "all", "metric": metric}]

        jobs += [dict(**j, **nj) for j in job for nj in new_job]

    return jobs


def get_medium_and_hard_datasets(scenario: str):
    prior_df = pd.read_csv("benchmark_data/gt_prior_data/origin_table.csv")
    medium_df = prior_df[((prior_df["difficulty"] == "medium") | (prior_df["difficulty"] == "hard")) & (prior_df["scenario"] == scenario)]  #
    medium_df = medium_df[medium_df["prior_trace"].notna()]
    return medium_df["dataset"].unique().tolist()
