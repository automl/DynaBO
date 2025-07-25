import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, RandomFacade, Scenario
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.config_selector import ConfigSelector

from dynabo.smac_additions.dynamic_prior_callback import LogIncumbentCallback
from dynabo.utils.cluster_utils import initialise_experiments
from dynabo.utils.evaluator import MFPBenchEvaluator, YAHPOGymEvaluator, ask_tell_opt, get_yahpo_fixed_parameter_combinations

EXP_CONFIG_FILE_PATH = "dynabo/experiments/baseline_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    benchmarklib: str = config["benchmarklib"]

    # Yahpo Values
    scenario: str = config["scenario"]
    dataset: str = config["dataset"]
    metric: str = config["metric"]

    # Baselien or Random
    baseline: bool = bool(config["baseline"])
    random: bool = bool(config["random"])

    # SMAC Scenario Values
    timeout: int = int(config["timeout_total"])
    seed: int = int(config["seed"])
    n_trials = int(config["n_trials"])

    # Initial Design values
    n_configs_per_hyperparameter = int(config["n_configs_per_hyperparameter"])
    max_ratio = float(config["max_ratio"])

    if benchmarklib == "yahpogym":
        evaluator: YAHPOGymEvaluator = YAHPOGymEvaluator(
            scenario=scenario,
            dataset=dataset,
            metric=metric,
            runtime_metric_name="timetrain" if scenario != "lcbench" else "time",
        )
    elif benchmarklib == "mfpbench":
        evaluator: MFPBenchEvaluator = MFPBenchEvaluator(
            scenario=scenario,
            seed=seed,
        )

    configuration_space = evaluator.get_configuration_space()

    smac_scenario = Scenario(configspace=configuration_space, deterministic=True, seed=seed, n_trials=n_trials)

    incumbent_callback = LogIncumbentCallback(
        result_processor=result_processor,
        evaluator=evaluator,
    )

    if baseline:
        initial_design_size = n_configs_per_hyperparameter * len(configuration_space)
        max_initial_design_size = int(max(1, min(initial_design_size, (max_ratio * smac_scenario.n_trials))))
        if initial_design_size != max_initial_design_size:
            initial_design_size = max_initial_design_size

        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario=smac_scenario, n_configs=initial_design_size)

        acquisition_function = HyperparameterOptimizationFacade.get_acquisition_function(scenario=scenario)

        local_and_prior_search = LocalAndSortedRandomSearch(
            configspace=configuration_space,
            acquisition_function=acquisition_function,
            max_steps=500,  # TODO wie viele local search steps sind reasonable?
        )
        config_selector = ConfigSelector(scenario=smac_scenario, max_new_config_tries=100)

        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario=smac_scenario,
            max_config_calls=1,
        )

        smac = HyperparameterOptimizationFacade(
            scenario=smac_scenario,
            target_function=evaluator.train,
            acquisition_function=acquisition_function,
            acquisition_maximizer=local_and_prior_search,
            config_selector=config_selector,
            callbacks=[incumbent_callback],
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )
    elif random:
        initial_design_size = None
        smac = RandomFacade(
            scenario=smac_scenario,
            target_function=evaluator.train,
            callbacks=[incumbent_callback],
            overwrite=True,
        )

    start_time = time.time()
    ask_tell_opt(smac=smac, evaluator=evaluator, timeout=timeout, result_processor=result_processor)
    end_time = time.time()

    optimization_data = evaluator.get_metadata()

    result = {
        "initial_design_size": initial_design_size,
        "final_performance": optimization_data["final_performance"],
        "runtime": round(end_time - start_time, 3),
        "virtual_runtime": optimization_data["virtual_runtime"],
        "reasoning_runtime": round(evaluator.reasoning_runtime, 3),
        "n_evaluations_computed": optimization_data["n_evaluations_computed"],
        "experiment_finished": True,
    }

    result_processor.process_results(results=result)


if __name__ == "__main__":
    initialise_experiments()
    benchmarklib = "mfpbench"
    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    smac_baseline = True
    random_baseline = False
    fill = True
    if fill:
        experimenter.fill_table_from_combination(
            parameters={
                "benchmarklib": [benchmarklib],
                "timeout_total": [86400],
                "n_trials": [50],
                "n_configs_per_hyperparameter": [10],
                "max_ratio": [0.25],
                "seed": range(30),
            },
            fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(
                benchmarklib=benchmarklib,
                acquisition_function="expected_improvement",
                with_all_datasets=False,
                medium_and_hard=True,
                pibo=False,
                dynabo=False,
                baseline=True,
                random=False,
                decay_enumerator=None,
                validate_prior=False,
                prior_validation_manwhitney=False,
                prior_validation_difference=False,
                n_prior_validation_samples=None,
                prior_validation_manwhitney_p=None,
                prior_validation_difference_threshold=None,
            ),
        )
    execute = False
    if execute:
        experimenter.execute(run_experiment, max_experiments=1, random_order=True)
