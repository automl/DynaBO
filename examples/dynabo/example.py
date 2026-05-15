"""Minimal DynaBO example: SMAC with dynamic prior injection on MFPBench, logged to SQLite."""

import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialInfo, TrialValue

from dynabo.smac_additions.dynamic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.smac_additions.dynamic_prior_callback import DynaBOWellPerformingPriorCallback, LogIncumbentCallback
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch
from dynabo.utils.evaluator import MFPBenchEvaluator

CONFIG_FILE = "examples/dynabo/config.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    evaluator = MFPBenchEvaluator(scenario=config["scenario"], seed=config["seed"])
    configuration_space = evaluator.get_configuration_space()

    smac_scenario = Scenario(
        configspace=configuration_space,
        deterministic=True,
        seed=config["seed"],
        n_trials=config["n_trials"],
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario=smac_scenario,
        n_configs=1,
    )

    acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(smac_scenario),
        initial_design_size=initial_design._n_configs,
        dynabo=True,
        prior_decay="linear",
    )

    local_and_prior_search = LocalAndPriorSearch(
        configspace=configuration_space,
        acquisition_function=acquisition_function,
        max_steps=500,
    )

    config_selector = ConfigSelector(scenario=smac_scenario, retries=100, retrain_after=1)

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario=smac_scenario,
        max_config_calls=1,
    )

    runhistory_encoder = BlackBoxFacade.get_runhistory_encoder(scenario=smac_scenario)

    prior_callback = DynaBOWellPerformingPriorCallback(
        # Prior type and position
        no_incumbent_percentile=0.01,
        prior_static_position=True,
        prior_every_n_trials=1,
        prior_chance_theta=0.015,
        prior_at_start=True,
        # Benchmark / data
        remove_old_prior=False,
        benchmarklib="mfpbench",
        scenario=evaluator.scenario,
        dataset=evaluator.dataset,
        metric="cost",
        base_path="benchmark_data/prior_data/",
        # Design and decay
        initial_design_size=initial_design._n_configs,
        prior_std_denominator=5,
        prior_decay_enumerator=5,
        prior_decay_denominator=1,
        # Validation (disabled for minimal example)
        validate_prior=False,
        prior_validation_method=None,
        n_prior_validation_samples=None,
        n_prior_based_samples=0,
        prior_validation_manwhitney_p_value=None,
        prior_validation_difference_threshold=None,
        # Logging
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
        runhistory_encoder=runhistory_encoder,
    )

    start_time = time.time()
    while smac.runhistory.finished < smac.scenario.n_trials:
        trial_info: TrialInfo = smac.ask()
        cost, runtime = evaluator.train(trial_info.config)
        smac.tell(trial_info, TrialValue(cost=cost, time=runtime))
    end_time = time.time()

    metadata = evaluator.get_metadata()
    result_processor.process_results(
        {
            "final_cost": metadata["final_cost"],
            "runtime": round(end_time - start_time, 3),
        }
    )


if __name__ == "__main__":
    experimenter = PyExperimenter(
        experiment_configuration_file_path=CONFIG_FILE,
        use_codecarbon=False,
    )

    experimenter.fill_table_from_combination(
        parameters={
            "scenario": ["lm1b_transformer_2048"],
            "seed": [0],
            "n_trials": [50],
        }
    )

    experimenter.execute(run_experiment, max_experiments=1)
