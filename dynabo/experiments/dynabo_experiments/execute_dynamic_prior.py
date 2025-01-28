import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.main.config_selector import ConfigSelector
from dynabo.smac_additions.dynamic_prior_callback import (
    DynaBODeceivingPriorCallback,
    DynaBOMediumPriorCallback,
    DynaBOMisleadingPriorCallback,
    DynaBOWellPerformingPriorCallback,
    LogIncumbentCallback,
)
from dynabo.smac_additions.dynmaic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch
from dynabo.utils.cluster_utils import intiialise_experiments
from dynabo.utils.yahpogym_evaluator import YAHPOGymEvaluator, get_yahpo_fixed_parameter_combinations

EXP_CONFIG_FILE_PATH = "dynabo/experiments/dynabo_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"


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
    initial_design_size = int(config["initial_design_size"])

    # Prior Values
    prior_kind = config["prior_kind"]
    prior_every_n_trials = int(config["prior_every_n_trials"])
    prior_std_denominator = float(config["prior_std_denominator"])
    validate_prior = config["validate_prior"]
    prior_p_value = float(config["prior_p_value"])
    n_prior_validation_samples = int(config["n_prior_validation_samples"])
    prior_sampling_weight = config["prior_sampling_weight"]

    evaluator: YAHPOGymEvaluator = YAHPOGymEvaluator(
        scenario=scenario,
        dataset=dataset,
        internal_timeout=internal_timeout,
        metric=metric,
        runtime_metric_name="timetrain" if scenario != "lcbench" else "time",
    )

    configuration_space = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)

    start_time = time.time()
    smac_scenario = Scenario(configspace=configuration_space, deterministic=True, seed=seed, n_trials=n_trials)

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

    if prior_kind == "good":
        PriorCallbackClass = DynaBOWellPerformingPriorCallback
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
        prior_every_n_iterations=prior_every_n_trials,
        prior_std_denominator=prior_std_denominator,
        prior_sampling_weight=prior_sampling_weight,
        n_prior_validation_samples=n_prior_validation_samples,
        prior_p_value=prior_p_value,
        initial_design_size=initial_design._n_configs,
        result_processor=result_processor,
        evaluator=evaluator,
        validate_prior=validate_prior,
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

    ask_tell_opt(smac=smac, evaluator=evaluator, timeout=timeout, result_processor=result_processor)
    end_time = time.time()

    result = {
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

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    fill = False
    if fill:
        experimenter.fill_table_from_combination(
            parameters={
                "benchmarklib": ["yahpogym"],
                "prior_kind": ["good", "medium", "misleading"],
                "prior_every_n_trials": [50],
                "validate_prior": [True, False],
                "n_prior_validation_samples": [500],
                "prior_p_value": [0.05],
                "prior_std_denominator": 5,
                "prior_sampling_weight": [0.3],
                "timeout_total": [86400],
                "timeout_internal": [1200],
                "n_trials": [200],
                "initial_design_size": [20],
                "seed": range(30),
            },
            fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(with_datasets=False, medium_and_hard=True),
        )
    reset = False
    if reset:
        experimenter.reset_experiments("error", "running")
    experimenter.execute(run_experiment, max_experiments=1, random_order=True)
