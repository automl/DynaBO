import time
from functools import partial

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, Scenario
from smac.main.config_selector import ConfigSelector

from dynabo.smac_additions.dynamic_prior_callback import (
    DynaBODeceivingPriorCallback,
    DynaBOMediumPriorCallback,
    DynaBOMisleadingPriorCallback,
    DynaBOWellPerformingPriorCallback,
    LogIncumbentCallback,
    PiBODeceivingPriorCallback,
    PiBOMediumPriorCallback,
    PiBOMisleadingPriorCallback,
    PiBOWellPerformingPriorCallback,
)
from dynabo.smac_additions.dynmaic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch
from dynabo.utils.cluster_utils import initialise_experiments
from dynabo.utils.evaluator import YAHPOGymEvaluator, ask_tell_opt, get_yahpo_fixed_parameter_combinations

EXP_CONFIG_FILE_PATH = "dynabo/experiments/prior_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    # Yahpo Values
    scenario: str = config["scenario"]
    dataset: str = config["dataset"]
    metric: str = config["metric"]

    # DynaBO or PIBO
    dynabo: bool = config["dynabo"]
    pibo: bool = config["pibo"]

    assert dynabo ^ pibo, "Either DynaBO or PiBO must be True"

    # SMAC Scenario Values
    timeout: int = int(config["timeout_total"])
    seed: int = int(config["seed"])
    n_trials = int(config["n_trials"])

    # Initial Design values
    n_configs_per_hyperparameter = int(config["n_configs_per_hyperparameter"])
    max_ratio = float(config["max_ratio"])

    # Prior Values
    prior_kind = config["prior_kind"]
    prior_every_n_trials = int(config["prior_every_n_trials"])
    validate_prior = config["validate_prior"]
    n_prior_validation_samples = int(config["n_prior_validation_samples"])
    prior_validation_p_value = float(config["prior_validation_p_value"])
    prior_std_denominator = float(config["prior_std_denominator"])
    prior_decay_enumerator = float(config["prior_decay_enumerator"])
    prior_decay_denominator = float(config["prior_decay_denominator"])

    # Extract Information of what happens in case of no incumbent
    no_incumbent_percentile = config["no_incumbent_percentile"]

    exponential_prior = config["exponential_prior"]
    prior_sampling_weight = config["prior_sampling_weight"]

    evaluator: YAHPOGymEvaluator = YAHPOGymEvaluator(
        scenario=scenario,
        dataset=dataset,
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

    acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(smac_scenario),
        initial_design_size=initial_design._n_configs,
    )

    local_and_prior_search = LocalAndPriorSearch(
        configspace=configuration_space,
        acquisition_function=acquisition_function,
        max_steps=500,  # TODO wie viele local search steps sind reasonable?
    )
    config_selector = ConfigSelector(scenario=smac_scenario, retries=100)

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario=smac_scenario,
        max_config_calls=1,
    )

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario=smac_scenario,
        max_config_calls=1,
    )

    if pibo:
        if prior_kind == "good":
            PriorCallbackClass = partial(PiBOWellPerformingPriorCallback, no_incumbent_percentile=no_incumbent_percentile)
        elif prior_kind == "medium":
            PriorCallbackClass = PiBOMediumPriorCallback
        elif prior_kind == "misleading":
            PriorCallbackClass = PiBOMisleadingPriorCallback
        elif prior_kind == "deceiving":
            PriorCallbackClass = PiBODeceivingPriorCallback
        else:
            raise ValueError(f"Prior kind {prior_kind} not supported")
    elif dynabo:
        if prior_kind == "good":
            PriorCallbackClass = partial(DynaBOWellPerformingPriorCallback, no_incumbent_percentile=no_incumbent_percentile)
        elif prior_kind == "medium":
            PriorCallbackClass = DynaBOMediumPriorCallback
        elif prior_kind == "misleading":
            PriorCallbackClass = DynaBOMisleadingPriorCallback
        elif prior_kind == "deceiving":
            PriorCallbackClass = DynaBODeceivingPriorCallback
        else:
            raise ValueError(f"Prior kind {prior_kind} not supported")

    prior_callback = PriorCallbackClass(
        scenario=evaluator.scenario,
        dataset=evaluator.dataset,
        metric=metric,
        base_path="benchmark_data/prior_data",
        initial_design_size=initial_design._n_configs,
        prior_every_n_trials=prior_every_n_trials,
        validate_prior=validate_prior,
        n_prior_validation_samples=n_prior_validation_samples,
        prior_validation_p_value=prior_validation_p_value,
        prior_std_denominator=prior_std_denominator,
        prior_decay_enumerator=prior_decay_enumerator,
        prior_decay_denominator=prior_decay_denominator,
        exponential_prior=exponential_prior,
        prior_sampling_weight=prior_sampling_weight,
        result_processor=result_processor,
        evaluator=evaluator,
    )

    incumbent_callback = LogIncumbentCallback(result_processor=result_processor, evaluator=evaluator)

    smac = HyperparameterOptimizationFacade(
        scenario=smac_scenario,
        target_function=evaluator.train,
        acquisition_function=acquisition_function,
        acquisition_maximizer=local_and_prior_search,
        config_selector=config_selector,
        callbacks=[prior_callback, incumbent_callback],
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

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    fill = True
    if fill:
        experimenter.fill_table_from_combination(
            parameters={
                "benchmarklib": ["yahpogym"],
                "prior_kind": ["good", "medium", "misleading"],
                "prior_every_n_trials": [50],
                "validate_prior": [False],
                "n_prior_validation_samples": [500],
                "prior_validation_p_value": [0.05],
                "prior_std_denominator": 5,
                "prior_decay_denominator": [10],
                "exponential_prior": [False],
                "prior_sampling_weight": [0.3],
                "no_incumbent_percentile": [0.01],
                "timeout_total": [86400],
                "n_trials": [200],
                "n_configs_per_hyperparameter": [10],
                "max_ratio": [0.25],
                "seed": range(10),
            },
            fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(with_all_datasets=False, medium_and_hard=True, pibo=True, dynabo=True, acquisition_function="expected_improvement"),
        )
    reset = False
    if reset:
        experimenter.reset_experiments("running", "error")
    execute = False
    if execute:
        experimenter.execute(run_experiment, max_experiments=20, random_order=True)
