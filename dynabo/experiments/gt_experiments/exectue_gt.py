import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.config_selector import ConfigSelector

from dynabo.smac_additions.dynamic_prior_callback import LogIncumbentCallback
from dynabo.utils.cluster_utils import intiialise_experiments
from dynabo.utils.yahpogym_evaluator import YAHPOGymEvaluator, ask_tell_opt, get_yahpo_fixed_parameter_combinations

EXP_CONFIG_FILE_PATH = "dynabo/experiments/gt_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    # Yahpo Values
    scenario: str = config["scenario"]
    dataset: str = config["dataset"]
    metric: str = config["metric"]

    # SMAC Scenario Values
    internal_timeout: int = int(config["timeout_internal"])
    timeout: int = int(config["timeout_total"])
    seed: int = int(config["seed"])
    n_trials = int(config["n_trials"])

    # Initial Design values
    n_configs_per_hyperparameter = int(config["n_configs_per_hyperparameter"])
    max_ratio = float(config["max_ratio"])

    evaluator: YAHPOGymEvaluator = YAHPOGymEvaluator(
        scenario=scenario,
        dataset=dataset,
        internal_timeout=internal_timeout,
        metric=metric,
        runtime_metric_name="timetrain" if scenario != "lcbench" else "time",
    )

    configuration_space = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)

    smac_scenario = Scenario(configspace=configuration_space, deterministic=True, seed=seed, n_trials=n_trials)

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
    config_selector = ConfigSelector(scenario=smac_scenario, retries=100)

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario=smac_scenario,
        max_config_calls=1,
    )

    incumbent_callback = LogIncumbentCallback(
        result_processor=result_processor,
        evaluator=evaluator,
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

    result = {
        "initial_design_size": initial_design_size,
        "final_performance": (-1) * evaluator.incumbent_cost,
        "runtime": round(end_time - start_time, 3),
        "virtual_runtime": round(evaluator.accumulated_runtime + evaluator.reasoning_runtime, 3),
        "reasoning_runtime": round(evaluator.reasoning_runtime, 3),
        "n_evaluations_computed": evaluator.eval_counter,
        "n_timeouts_occurred": evaluator.timeout_counter,
        "experiment_finished": True,
    }

    result_processor.process_results(results=result)


if __name__ == "__main__":
    intiialise_experiments()

    base_parameters = {
        "benchmarklib": ["yahpogym"],
        "prior_kind": ["good"],
        "prior_every_n_trials": [50],
        "validate_prior": [False],
        "prior_std_denominator": 5,
        "timeout_total": [86400],
        "timeout_internal": [1200],
        "n_trials": [5000],
        "n_configs_per_hyperparameter": [10],
        "max_ratio": [0.25],
    }
    all_one_seed = False
    if all_one_seed:
        experimenter = PyExperimenter(
            experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
            database_credential_file_path=DB_CRED_FILE_PATH,
            use_codecarbon=False,
        )

    medium_and_hard = True
    if medium_and_hard:
        experimenter = PyExperimenter(
            experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
            database_credential_file_path=DB_CRED_FILE_PATH,
            use_codecarbon=False,
            table_name="data_generation_medium_hard",
        )

    fill_table = False
    if fill_table:
        if all_one_seed:
            experimenter.fill_table_from_combination(
                parameters={**base_parameters, **{"seed": range(1)}},
                fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(with_all_datasets=True, medium_and_hard=False, baseline=True),
            )
        if medium_and_hard:
            experimenter.fill_table_from_combination(
                parameters={**base_parameters, **{"seed": range(10)}},
                fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(with_all_datasets=False, medium_and_hard=True, baseline=True),
            )
    execute = True
    if execute:
        experimenter.execute(run_experiment, max_experiments=1)
