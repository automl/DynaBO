from ConfigSpace import Configuration

from dynabo.utils.jahsbench201_evaluator import JAHSBENCH201_DATASET_OPTIONS, JAHSBench201Evaluator

if __name__ == '__main__':
    for i in range(len(JAHSBENCH201_DATASET_OPTIONS)):
        print(JAHSBENCH201_DATASET_OPTIONS[i])
        evaluator: JAHSBench201Evaluator = JAHSBench201Evaluator(dataset=JAHSBENCH201_DATASET_OPTIONS[i])

        cs = evaluator.get_configuration_space()
        print(cs)
        for j in range(100):
            config: Configuration = cs.sample_configuration()
            print("juhu")
            config["Optimizer"] = "SGD"
            print(config)

            print(" ")
            print(" ")
            print(" ")
            print(" ")
            print(evaluator.train(config))
        break



