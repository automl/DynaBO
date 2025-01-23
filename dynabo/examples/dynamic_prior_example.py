from typing import Dict

from ConfigSpace import (
    ConfigurationSpace,
)
from matplotlib import pyplot as plt
from smac.facade import HyperparameterOptimizationFacade
from smac.scenario import Scenario

from dynabo.experiments.gt_experiments.execute_gt import YAHPOGymEvaluator
from dynabo.smac_additions.dynamic_prior_callback import DynaBOWellPerformingPriorCallback
from dynabo.smac_additions.dynmaic_prior_acquisition_function import DynamicPriorAcquisitionFunction
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch


def plot(facades: Dict[str, HyperparameterOptimizationFacade]):
    plt.figure(figsize=(10, 5))
    for facade_name, facade in facades.items():
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            X.append(item.trial)
            Y.append(item.costs[0])
        plt.step(X, Y, label=facade_name, where="post")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    facades = {}
    # Prior optimization
    seed = 42

    evaluator = YAHPOGymEvaluator(
        scenario="rbv2_ranger",
        dataset="1220",
        metric="acc",
    )

    prior_configuration_space: ConfigurationSpace = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)

    prior_scenario = Scenario(prior_configuration_space, n_trials=200, seed=seed)

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario=prior_scenario)

    # We define the prior acquisition function, which conduct the optimization using priors over the optimum
    prior_acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(prior_scenario),
        initial_design_size=initial_design._n_configs,
    )

    prior_intensifier = HyperparameterOptimizationFacade.get_intensifier(
        prior_scenario,
        max_config_calls=1,
    )

    prior_callback = DynaBOWellPerformingPriorCallback(
        scenario=evaluator.scenario,
        dataset=evaluator.dataset,
        metric="acc",
        base_path="benchmark_data/prior_data",
        prior_every_n_iterations=20,
        initial_design_size=initial_design._n_configs,
    )

    # Create our SMAC object and pass the scenario and the train method
    prior_smac = HyperparameterOptimizationFacade(
        prior_scenario,
        evaluator.train,
        acquisition_function=prior_acquisition_function,
        acquisition_maximizer=LocalAndPriorSearch(
            configspace=prior_configuration_space,
            acquisition_function=prior_acquisition_function,
            max_steps=100,
        ),
        callbacks=[
            prior_callback,
        ],
        initial_design=initial_design,
        intensifier=prior_intensifier,
        overwrite=True,
    )
    prior_incumbent = prior_smac.optimize()
    facades = {"prior": prior_smac}

    # Base optimization
    seed = 42

    evaluator = YAHPOGymEvaluator(
        scenario="rbv2_ranger",
        dataset="1220",
        metric="acc",
    )

    base_configuration_space: ConfigurationSpace = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)
    base_scenario = Scenario(base_configuration_space, n_trials=200, seed=seed)

    base_intensifier = HyperparameterOptimizationFacade.get_intensifier(
        base_scenario,
        max_config_calls=1,
    )

    # Create our SMAC object and pass the scenario and the train method
    base_smac = HyperparameterOptimizationFacade(
        base_scenario,
        evaluator.train,
        intensifier=base_intensifier,
        overwrite=True,
    )
    base_incumbent = base_smac.optimize()
    facades["base"] = base_smac

    plot(facades)
