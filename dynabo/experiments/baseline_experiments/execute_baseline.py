import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from smac import HyperparameterOptimizationFacade, RandomFacade, Scenario, BlackBoxFacade
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.config_selector import ConfigSelector

from ConfigSpace import Constant, ConfigurationSpace

from dynabo.smac_additions.dynamic_prior_callback import LogIncumbentCallback
from dynabo.utils.cluster_utils import initialise_experiments
from dynabo.utils.configuration_data_classes import (
    BenchmarkConfig,
    InitialDesignConfig,
    SMACConfig,
)
from dynabo.utils.evaluator import MFPBenchEvaluator, YAHPOGymEvaluator, ask_tell_opt, fill_table

EXP_CONFIG_FILE_PATH = "dynabo/experiments/baseline_experiments/config.yml"
DB_CRED_FILE_PATH = "config/database_credentials.yml"

only_two_hyperparameters = False


def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    # Extract all configurations
    benchmark_cfg = BenchmarkConfig.from_config(config)
    baseline = bool(config["baseline"])
    random = bool(config["random"])
    smac_cfg = SMACConfig.from_config(config)
    initial_design_cfg = InitialDesignConfig.from_config(config)

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

    if only_two_hyperparameters:
        # Remove opt_momentum and lr_power from configuration space
        learning_rate = configuration_space.get("lr_initial")
        lr_decay_factor = configuration_space.get("lr_decay_factor")
        # Add opt_momentum and lr_power as static hyperparameters
        if benchmark_cfg.scenario == "cifar100_wideresnet_2048":
            lr_power = 0.1520241567779
            opt_momentum = 0.9516762484947
        elif benchmark_cfg.scenario == "imagenet_resnet_512":
            lr_power = 1.696464843003
            opt_momentum = 0.9106070776233
        elif benchmark_cfg.scenario == "lm1b_transformer_2048":
            lr_power = 0.8995646537353
            opt_momentum = 0.9980083177484
        elif benchmark_cfg.scenario == "translatewmt_xformer_64":
            lr_power = 0.1249360373907
            opt_momentum = 0.9943414718914
        else:
            raise ValueError(f"Scenario {benchmark_cfg.scenario} not found")

        new_configuration_space = ConfigurationSpace()
        new_configuration_space.add(learning_rate)
        new_configuration_space.add(lr_decay_factor)
        new_configuration_space.add(Constant("opt_momentum", opt_momentum))
        new_configuration_space.add(Constant("lr_power", lr_power))

    smac_scenario = Scenario(configspace=configuration_space, deterministic=True, seed=smac_cfg.seed, n_trials=smac_cfg.n_trials)

    incumbent_callback = LogIncumbentCallback(
        result_processor=result_processor,
        evaluator=evaluator,
    )

    if baseline:
        initial_design_size = initial_design_cfg.n_configs_per_hyperparameter * len(configuration_space)
        max_initial_design_size = int(max(1, min(initial_design_size, (initial_design_cfg.max_ratio * smac_scenario.n_trials))))
        if initial_design_size != max_initial_design_size:
            initial_design_size = max_initial_design_size

        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario=smac_scenario, n_configs=initial_design_size)

        acquisition_function = smac_cfg.get_acquisition_function()

        local_and_prior_search = LocalAndSortedRandomSearch(
            configspace=configuration_space,
            acquisition_function=acquisition_function,
            max_steps=500,  # TODO wie viele local search steps sind reasonable?
        )
        config_selector = ConfigSelector(scenario=smac_scenario, retries=100, retrain_after=1)

        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario=smac_scenario,
            max_config_calls=1,
        )

        runhistory_encoder = BlackBoxFacade.get_runhistory_encoder(scenario=smac_scenario)

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
            runhistory_encoder=runhistory_encoder,
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
    benchmarklib = "mfpbench"
    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )
    fill = True
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
            approach="baseline",
            approach_parameters=None,
        )
    reset = False
    if reset:
        experimenter.reset_experiments("error", "running")

    execute = False
    if execute:
        experimenter.execute(run_experiment, max_experiments=1, random_order=True)
