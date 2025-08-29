import time
from functools import partial

from numpy import True_
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
from dynabo.utils.configuration_data_classes import (
    BenchmarkConfig,
    InitialDesignConfig,
    PriorConfig,
    PriorDecayConfig,
    PriorValidationConfig,
    SMACConfig,
    extract_optimization_approach,
)
from dynabo.utils.evaluator import MFPBenchEvaluator, YAHPOGymEvaluator, ask_tell_opt, fill_table

EXP_CONFIG_FILE_PATH = "dynabo/experiments/prior_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    # Extract all configurations
    benchmark_cfg = BenchmarkConfig.from_config(config)
    dynabo, pibo = extract_optimization_approach(config)
    smac_cfg = SMACConfig.from_config(config)
    initial_design_cfg = InitialDesignConfig.from_config(config)
    prior_cfg = PriorConfig.from_config(config)
    prior_decay_cfg = PriorDecayConfig.from_config(config)
    prior_validation_cfg = PriorValidationConfig.from_config(config)

    if benchmark_cfg.benchmarklib == "yahpogym":
        evaluator: YAHPOGymEvaluator = YAHPOGymEvaluator(
            scenario=benchmark_cfg.scenario,
            dataset=benchmark_cfg.dataset,
            metric=benchmark_cfg.metric,
            runtime_metric_name="timetrain" if benchmark_cfg.scenario != "lcbench" else "time",
        )
    elif benchmark_cfg.benchmarklib == "mfpbench":
        evaluator: MFPBenchEvaluator = MFPBenchEvaluator(
            scenario=benchmark_cfg.scenario,
            seed=smac_cfg.seed,
        )

    configuration_space = evaluator.get_configuration_space()

    smac_scenario = Scenario(configspace=configuration_space, deterministic=True, seed=smac_cfg.seed, n_trials=smac_cfg.n_trials)

    initial_design_size = initial_design_cfg.n_configs_per_hyperparameter * len(configuration_space)
    max_initial_design_size = int(max(1, min(initial_design_size, (initial_design_cfg.max_ratio * smac_scenario.n_trials))))
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

    if pibo:
        if prior_cfg.kind == "good":
            PriorCallbackClass = partial(PiBOWellPerformingPriorCallback, no_incumbent_percentile=prior_cfg.no_incumbent_percentile)
        elif prior_cfg.kind == "medium":
            PriorCallbackClass = partial(PiBOMediumPriorCallback, no_incumbent_percentile=prior_cfg.no_incumbent_percentile)
        elif prior_cfg.kind == "misleading":
            PriorCallbackClass = PiBOMisleadingPriorCallback
        elif prior_cfg.kind == "deceiving":
            PriorCallbackClass = PiBODeceivingPriorCallback
        else:
            raise ValueError(f"Prior kind {prior_cfg.kind} not supported")
    elif dynabo:
        if prior_cfg.kind == "good":
            PriorCallbackClass = partial(
                DynaBOWellPerformingPriorCallback,
                no_incumbent_percentile=prior_cfg.no_incumbent_percentile,
                prior_static_position=prior_cfg.prior_static_position,
                prior_every_n_trials=prior_cfg.prior_every_n_trials,
                prior_chance_theta=prior_cfg.chance_theta,
                prior_at_start=prior_cfg.at_start,
            )
        elif prior_cfg.kind == "medium":
            PriorCallbackClass = partial(
                DynaBOMediumPriorCallback,
                no_incumbent_percentile=prior_cfg.no_incumbent_percentile,
                prior_static_position=prior_cfg.prior_static_position,
                prior_every_n_trials=prior_cfg.prior_every_n_trials,
                prior_chance_theta=prior_cfg.chance_theta,
                prior_at_start=prior_cfg.at_start,
            )
        elif prior_cfg.kind == "misleading":
            PriorCallbackClass = partial(
                DynaBOMisleadingPriorCallback,
                prior_static_position=prior_cfg.prior_static_position,
                prior_every_n_trials=prior_cfg.prior_every_n_trials,
                prior_chance_theta=prior_cfg.chance_theta,
                prior_at_start=prior_cfg.at_start,
            )
        elif prior_cfg.kind == "deceiving":
            PriorCallbackClass = partial(
                DynaBODeceivingPriorCallback,
                prior_static_position=prior_cfg.prior_static_position,
                prior_every_n_trials=prior_cfg.prior_every_n_trials,
                prior_chance_theta=prior_cfg.chance_theta,
                prior_at_start=prior_cfg.at_start,
            )
        else:
            raise ValueError(f"Prior kind {prior_cfg.kind} not supported")

    prior_callback = PriorCallbackClass(
        benchmarklib=benchmark_cfg.benchmarklib,
        scenario=evaluator.scenario,
        dataset=evaluator.dataset,
        metric=benchmark_cfg.metric,
        base_path="benchmark_data/prior_data/",
        initial_design_size=initial_design._n_configs,
        validate_prior=prior_validation_cfg.validate,
        prior_validation_method=prior_validation_cfg.method,
        n_prior_validation_samples=prior_validation_cfg.n_samples,
        prior_validation_manwhitney_p_value=prior_validation_cfg.manwhitney_p_value,
        prior_validation_difference_threshold=prior_validation_cfg.difference_threshold,
        prior_std_denominator=prior_cfg.std_denominator,
        prior_decay_enumerator=prior_decay_cfg.enumerator,
        prior_decay_denominator=prior_decay_cfg.denominator,
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
    ask_tell_opt(smac=smac, evaluator=evaluator, timeout=smac_cfg.timeout, result_processor=result_processor)
    end_time = time.time()

    optimization_data = evaluator.get_metadata()

    result = {
        "initial_design_size": initial_design_size,
        "final_cost": optimization_data["final_cost"],
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
    benchmarklib = "mfpbench"
    fill = False

    if fill:
        fill_table(
            py_experimenter=experimenter,
            common_parameters={
                "acquisition_function": ["expected_improvement"],
                "timeout_total": [3600],
                "n_trials": [50],
                "initial_design__n_configs_per_hyperparameter": [10],
                "initial_design__max_ratio": [0.25],
                "seed": list(range(30)),
            },
            benchmarklib=benchmarklib,
            benchmark_parameters={
                "with_all_datasets": False,
                "medium_and_hard": True,
            },
            approach="dynabo",
            approach_parameters={
                # Prior configurationz
                "prior_kind_choices": ["good", "medium", "misleading", "deceiving"],
                "no_incumbent_percentile": 0.01,
                "prior_std_denominator": 5,
                # Dynabo when prior
                "prior_static_position": True,
                "prior_every_n_trials_choices": [10],
                "prior_at_start_choices": [True, False],
                "prior_chance_theta_choices": [0.01, 0.015],
                # Decay parameters
                "prior_decay_enumerator_choices": [50,],
                "prior_decay_denominator": 10,
                # Validation parameters
                "validate_prior_choices": [True, False],
                "prior_validation_method_choices": ["baseline_perfect", "difference"],
                "n_prior_validation_samples": 500,
                "prior_validation_manwhitney_p_choices": [0.05],
                "prior_validation_difference_threshold_choices": [-1],
            },
        )
    reset = False
    if reset:
        experimenter.reset_experiments("running", "error")
    execute = True
    if execute:
        experimenter.execute(run_experiment, max_experiments=1, random_order=True)
