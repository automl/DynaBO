import json

from omegaconf import OmegaConf
from carps.utils.running import make_task
from carps.utils.trials import TrialInfo

from ConfigSpace import Configuration, ConfigurationSpace
from py_experimenter.result_processor import ResultProcessor

MFPBENCH_SCENARIO_OPTIONS = ["cifar100_wideresnet_2048", "imagenet_resnet_512", "lm1b_transformer_2048",
                             "translatewmt_xformer_64"]

class MFPBenchEvaluator:
    def __init__(
        self,
        scenario,
        seed: int = 42,
        internal_timeout=-1,
        result_processor: ResultProcessor = None,
    ):
        self.scenario = scenario
        self.result_processor = result_processor
        self.internal_timeout = internal_timeout

        # setup mfpbench config
        exp_config = OmegaConf.load("CARP-S/carps/configs/task/MFPBench/SO/pd1/" + self.scenario + ".yaml")
        exp_config.seed = seed
        task = make_task(exp_config)

        self.task = task
        self.config_space = task.objective_function.configspace
        self.objective_function = task.objective_function.evaluate

        self.accumulated_runtime = 0
        self.reasoning_runtime = 0
        self.incumbent_trace = list()
        self.incumbent_cost = None
        self.eval_counter = 0
        self.timeout_counter = 0

    def train(self, config: Configuration, seed: int = 0):
        if self.eval_counter % 100 == 0 and self.result_processor is not None:
            from datetime import datetime
            now = datetime.now()
            self.result_processor.process_results({"num_evaluations": str(self.eval_counter) + " " + now.strftime("%m/%d/%Y, %H:%M:%S")})

        ti = TrialInfo(config=config, budget=1.0, seed=seed)
        res = self.objective_function(ti)

        performance = round(float(res.cost), 6)
        runtime = round(float(res.virtual_time), 3)

        # check whether internal evaluation timeout is set
        if self.internal_timeout != -1:
            # check whether timeout is hit
            if runtime > self.internal_timeout:
                self.accumulated_runtime += self.internal_timeout
                self.timeout_counter += 1
                raise Exception("Internal timeout exceeded")

        self.accumulated_runtime = round(self.accumulated_runtime + runtime, 3)

        if self.incumbent_cost is None or performance < self.incumbent_cost:
            self.incumbent_cost = performance
            incumbent_tuple = (
                round(self.reasoning_runtime + self.accumulated_runtime, 3),
                1 - performance,
                self.eval_counter,
                dict(config),
            )
            print("new incumbent:", incumbent_tuple)
            self.incumbent_trace.append(incumbent_tuple)
            if self.result_processor is not None:
                self.result_processor.process_results(
                    {
                        "incumbent_trace": json.dumps(self.incumbent_trace),
                    }
                )

        return float(performance), float(runtime)

    def get_configuration_space(self) -> ConfigurationSpace:
        return self.config_space
