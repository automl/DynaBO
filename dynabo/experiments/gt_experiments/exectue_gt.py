import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, RandomFacade, Scenario
from smac.acquisition.function import LCB
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.config_selector import ConfigSelector

from dynabo.smac_additions.dynamic_prior_callback import LogIncumbentCallback
from dynabo.utils.cluster_utils import initialise_experiments
from dynabo.utils.evaluator import MFPBenchEvaluator, YAHPOGymEvaluator, ask_tell_opt, get_yahpo_fixed_parameter_combinations

EXP_CONFIG_FILE_PATH = "dynabo/experiments/gt_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    benchmarklib: str = config["benchmarklib"]

    # Yahpo Values
    scenario: str = config["scenario"]
    dataset: str = config["dataset"]
    metric: str = config["metric"]

    # Check whether we use random facade
    random: bool = bool(config["random"])

    # SMAC Scenario Values
    timeout: int = int(config["timeout_total"])
    seed: int = int(config["seed"])
    n_trials = int(config["n_trials"])
    acquisition_function = config["acquisition_function"]

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

    if random:
        initial_design_size = None
        smac = RandomFacade(scenario=smac_scenario, target_function=evaluator.train, callbacks=[incumbent_callback], overwrite=True)

    else:
        initial_design_size = n_configs_per_hyperparameter * len(configuration_space)
        max_initial_design_size = int(max(1, min(initial_design_size, (max_ratio * smac_scenario.n_trials))))
        if initial_design_size != max_initial_design_size:
            initial_design_size = max_initial_design_size

        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario=smac_scenario, n_configs=initial_design_size)

        if acquisition_function == "expected_improvement":
            acquisition_function = HyperparameterOptimizationFacade.get_acquisition_function(scenario=scenario)
        elif acquisition_function == "confidence_bound":
            acquisition_function = LCB()
        else:
            raise ValueError(f"Acquisition function {acquisition_function} not known.")

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

    start_time = time.time()
    ask_tell_opt(smac=smac, evaluator=evaluator, timeout=timeout, result_processor=result_processor)
    end_time = time.time()

    optimization_data = evaluator.get_metadata()

    result = {
        "initial_design_size": initial_design_size,
        "runtime": round(end_time - start_time, 3),
        "reasoning_runtime": optimization_data["reasoning_runtime"],
        "final_performance": optimization_data["final_performance"],
        "virtual_runtime": optimization_data["virtual_runtime"],
        "n_evaluations_computed": optimization_data["n_evaluations_computed"],
        "experiment_finished": True,
    }

    result_processor.process_results(results=result)


if __name__ == "__main__":
    initialise_experiments()

    benchmarklib = "mfpbench"
    base_parameters = {
        "benchmarklib": [benchmarklib],
        "inverted_cost": [True],
        "timeout_total": [86400],
        "timeout_internal": [1200],
        "n_trials": [100],
        "n_configs_per_hyperparameter": [10],
        "max_ratio": [0.25],
    }
    all_one_seed = False
    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )

    medium_and_hard = True
    fill_table = True
    if fill_table:
        if all_one_seed:
            experimenter.fill_table_from_combination(
                parameters={**base_parameters, **{"seed": range(1)}},
                fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(
                    with_all_datasets=True, medium_and_hard=False, baseline=True, random=False, acquisition_function="confidence_bound"
                ),
            )
        if medium_and_hard:
            experimenter.fill_table_from_combination(
                parameters={**base_parameters, **{"seed": range(10)}},
                fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(
                    benchmarklib=benchmarklib, pibo=False, dynabo=False, baseline=True, with_all_datasets=False, medium_and_hard=True, random=False, acquisition_function="expected_improvement"
                ),
            )
    reset = False
    if reset:
        experimenter.reset_experiments("error")

    execute = True
    if execute:
        experimenter.execute(run_experiment, max_experiments=1)
