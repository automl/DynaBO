"""Minimal baseline example: SMAC hyperparameter optimization on MFPBench, logged to SQLite."""
import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.runhistory import TrialInfo, TrialValue

from dynabo.utils.evaluator import MFPBenchEvaluator

CONFIG_FILE = "examples/baseline/config.yml"


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    evaluator = MFPBenchEvaluator(scenario=config["scenario"], seed=config["seed"])

    smac_scenario = Scenario(
        configspace=evaluator.get_configuration_space(),
        deterministic=True,
        seed=config["seed"],
        n_trials=config["n_trials"],
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario=smac_scenario,
        n_configs=1,
    )

    smac = HyperparameterOptimizationFacade(
        scenario=smac_scenario,
        target_function=evaluator.train,
        initial_design=initial_design,
        overwrite=True,
    )

    start_time = time.time()
    while smac.runhistory.finished < smac.scenario.n_trials:
        trial_info: TrialInfo = smac.ask()
        cost, runtime = evaluator.train(trial_info.config)
        smac.tell(trial_info, TrialValue(cost=cost, time=runtime))
    end_time = time.time()

    metadata = evaluator.get_metadata()
    result_processor.process_results({
        "final_cost": metadata["final_cost"],
        "runtime": round(end_time - start_time, 3),
    })


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
