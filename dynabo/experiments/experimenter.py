import argparse
import json
import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import StatusType, TrialInfo, TrialValue
from yahpo_gym import benchmark_set, local_config

from dynabo.utils.yahpogym_evaluator import YAHPOGymEvaluator

EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def get_yahpo_fixed_parameter_combinations(with_datasets: bool = True):
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

        if with_datasets:
            # create ablation and ds_tunability jobs
            jobs += [{"scenario": scenario, "dataset": dataset, "metric": metric} for dataset in bench.instances]
        else:
            jobs += [{"scenario": scenario, "dataset": "all", "metric": metric}]
    return jobs


def ask_tell_opt(smac: HyperparameterOptimizationFacade, evaluator, timeout: int = 86400):
    while (evaluator.accumulated_runtime + evaluator.reasoning_runtime) < timeout:
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
        smac.tell(info=trial_info, value=trial_value, save=False)
        end_tell = time.time()

        # add runtime for tell
        tell_runtime = round(end_tell - start_tell, 3)
        evaluator.reasoning_runtime += tell_runtime


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    algo: str = config["algorithm"]
    scenario: str = config["scenario"]
    dataset: str = config["dataset"]
    metric: str = config["metric"]
    internal_timeout: int = int(config["timeout_internal"])
    timeout: int = int(config["timeout_total"])
    seed: int = int(config["seed"])

    eval: YAHPOGymEvaluator = YAHPOGymEvaluator(
        scenario=scenario,
        dataset=dataset,
        internal_timeout=internal_timeout,
        metric=metric,
        result_processor=result_processor,
    )

    start_time = time.time()
    if algo == "dynabo":
        pass
    elif algo == "pibo":
        pass
    elif algo == "priorband":
        pass
    elif algo == "groundtruth":
        smac_scenario = Scenario(
            configspace=eval.benchmark.get_opt_space(drop_fidelity_params=True),
            deterministic=True,
            seed=seed,
        )
        smac = HyperparameterOptimizationFacade(smac_scenario, eval.train)
        ask_tell_opt(smac=smac, evaluator=eval, timeout=timeout)
    end_time = time.time()

    result = {
        "actual_runtime": round(end_time - start_time, 3),
        "acc_virtual_runtime": round(eval.accumulated_runtime + eval.reasoning_runtime, 3),
        "acc_reasoning_runtime": round(eval.reasoning_runtime, 3),
        "final_performance": (-1) * eval.incumbent_cost,
        "incumbent_trace": json.dumps(eval.incumbent_trace),
        "num_evaluations": eval.eval_counter,
        "num_timeouts": eval.timeout_counter,
        "done": "true",
    }

    result_processor.process_results(results=result)


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path("yahpodata")

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    parser = argparse.ArgumentParser(
        prog="DynaBO Experimenter",
        description="This is the benchmark executor for the DynaBO paper.",
    )

    parser.add_argument("-s", "--setup", action="store_true", required=False)
    parser.add_argument(
        "-e",
        "--exec",
        action="store",
        help="Run the benchmark executor for a certain number of experiments.",
        required=False,
        default=0,
    )

    args = parser.parse_args()
    if args.setup:
        experimenter.fill_table_from_combination(
            parameters={
                "benchmarklib": ["yahpogym"],
                "timeout_internal": [300],
                "timeout_total": [3600],
                "algorithm": ["groundtruth"],
                "seed": range(2),
            },
            fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(),
        )

    if args.exec != 0:
        experimenter.execute(
            experiment_function=run_experiment,
            max_experiments=int(args.exec),
        )
