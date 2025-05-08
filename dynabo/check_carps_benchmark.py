from ConfigSpace import Configuration

from dynabo.utils.evaluator import MFPBENCH_SCENARIO_OPTIONS, MFPBenchEvaluator

if __name__ == "__main__":
    for i in range(len(MFPBENCH_SCENARIO_OPTIONS)):
        print(MFPBENCH_SCENARIO_OPTIONS[i])
        evaluator: MFPBenchEvaluator = MFPBenchEvaluator(scenario=MFPBENCH_SCENARIO_OPTIONS[i])

        cs = evaluator.get_configuration_space()
        print(cs)
        for j in range(100):
            config: Configuration = cs.sample_configuration()
            print("juhu")
            print(config)

            print(" ")
            print(" ")
            print(" ")
            print(" ")
            print(evaluator.train(config))
        break
