from typing import Dict

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from matplotlib import pyplot as plt
from smac.facade import AbstractFacade, HyperparameterOptimizationFacade
from smac.scenario import Scenario

from dynabo.experiments.experimenter import YAHPOGymEvaluator
from dynabo.smac_additions.change_prior_callback import AbstractDynamicPriorCallback
from dynabo.smac_additions.dynmaic_prior_activation_function import DynamicPriorAcquisitionFunction
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch


class PriorYahpoGymEvaluator(YAHPOGymEvaluator):
    def get_configuration_space(self) -> ConfigurationSpace:
        return self.benchmark.get_opt_space(drop_fidelity_params=True)

    def get_prior_configuration_space(self, optimum: Dict[str, float]) -> ConfigurationSpace:
        configuration_space = self.get_configuration_space()
        configspace_name = configuration_space.name

        new_configuration_space = ConfigurationSpace(name=f"{configspace_name}_prior")

        for hyperparameter in configuration_space.values():
            if isinstance(hyperparameter, CategoricalHyperparameter):
                old_choices = hyperparameter.choices
                optimum_choice = optimum.get(hyperparameter.name, None)
                new_weights = [1 if choice == optimum_choice else 0 for choice in old_choices]
                new_hyperparameter = CategoricalHyperparameter(
                    name=hyperparameter.name,
                    choices=hyperparameter.choices,
                    default_value=hyperparameter.default_value,
                    weights=new_weights,
                )
            elif isinstance(hyperparameter, UniformFloatHyperparameter):
                optimum_choice = optimum.get(hyperparameter.name, (hyperparameter.lower + hyperparameter.upper) / 2)
                new_hyperparameter = NormalFloatHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    mu=optimum_choice,
                    sigma=(hyperparameter.upper - hyperparameter.lower) / 2,
                    log=hyperparameter.log,
                )
            elif isinstance(hyperparameter, UniformIntegerHyperparameter):
                optimum_choice = optimum.get(hyperparameter.name, int((hyperparameter.lower + hyperparameter.upper) / 2))
                new_hyperparameter = NormalIntegerHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    mu=optimum_choice,
                    sigma=(hyperparameter.upper - hyperparameter.lower) / 2,
                    log=hyperparameter.log,
                )
            elif isinstance(hyperparameter, Constant):
                new_hyperparameter = hyperparameter
            else:
                raise NotImplementedError(f"Hyperparameter {hyperparameter} not supported.")

            new_configuration_space.add(new_hyperparameter)
        return new_configuration_space


def plot_trajectory(facades: Dict[str, AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades["base_smac"].scenario.objectives)

    for (
        facade_name,
        facade,
    ) in facades.items():
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.trial

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade_name)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluator = PriorYahpoGymEvaluator(
        scenario="rbv2_ranger",
        dataset="41157",
        metric="acc",
    )

    base_configuration_space: ConfigurationSpace = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)
    base_prior_configuration_space = evaluator.get_prior_configuration_space(
        optimum={
            "min.node.size": 1,
            "mtry.power": 0.7469734607851907,
            "num.impute.selected.cpo": "impute.hist",
            "num.trees": 183,
            "repl": 6,
            "respect.unordered.factors": "ignore",
            "sample.fraction": 0.8625101083119422,
            "splitrule": "extratrees",
            "task_id": "41157",
            "trainsize": 0.525,
            "num.random.splits": 100,
        }
    )

    base_scenario = Scenario(base_configuration_space, n_trials=20)

    # We define the prior acquisition function, which conduct the optimization using priors over the optimum
    base_acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(base_scenario),
        prior_configspace=base_configuration_space,
        decay_beta=base_scenario.n_trials / 10,  # Proven solid value
    )

    base_intensifier = HyperparameterOptimizationFacade.get_intensifier(
        base_scenario,
        max_config_calls=1,
    )

    # Create our SMAC object and pass the scenario and the train method
    base_smac = HyperparameterOptimizationFacade(
        base_scenario,
        evaluator.train,
        acquisition_function=base_acquisition_function,
        acquisition_maximizer=LocalAndPriorSearch(
            configspace=base_configuration_space,
            acquisition_function=base_acquisition_function,
        ),
        intensifier=base_intensifier,
        overwrite=True,
    )
    incumbent = base_smac.optimize()

    prior_base_configuration_space: ConfigurationSpace = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)
    prior_prior_configuration_space = evaluator.get_prior_configuration_space(
        optimum={
            "min.node.size": 1,
            "mtry.power": 0.7469734607851907,
            "num.impute.selected.cpo": "impute.hist",
            "num.trees": 183,
            "repl": 6,
            "respect.unordered.factors": "ignore",
            "sample.fraction": 0.8625101083119422,
            "splitrule": "extratrees",
            "task_id": "41157",
            "trainsize": 0.525,
            "num.random.splits": 100,
        }
    )

    prior_scenario = Scenario(prior_base_configuration_space, n_trials=20)
    prior_acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(prior_scenario),
        prior_configspace=prior_prior_configuration_space,
        decay_beta=prior_scenario.n_trials / 10,  # Proven solid value
    )
    prior_intensifier = HyperparameterOptimizationFacade.get_intensifier(prior_scenario, max_config_calls=1)
    prior_smac = HyperparameterOptimizationFacade(
        prior_scenario,
        evaluator.train,
        acquisition_function=prior_acquisition_function,
        acquisition_maximizer=LocalAndPriorSearch(
            configspace=prior_prior_configuration_space,
            acquisition_function=prior_acquisition_function,
        ),
        intensifier=prior_intensifier,
        overwrite=True,
    )
    prior_acquisition_function.dynamic_init(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(prior_scenario),
        decay_beta=prior_scenario.n_trials / 10,
        prior_configspace=base_prior_configuration_space,
    )
    prior_smac._acquisition_maximizer.dynamic_init(base_prior_configuration_space)
    prior_smac.optimize()

    dynamic_prior_base_configuration_space: ConfigurationSpace = evaluator.benchmark.get_opt_space(drop_fidelity_params=True)
    dynamic_prior_prior_configuration_space = evaluator.get_prior_configuration_space(
        optimum={
            "min.node.size": 1,
            "mtry.power": 0.7469734607851907,
            "num.impute.selected.cpo": "impute.hist",
            "num.trees": 183,
            "repl": 6,
            "respect.unordered.factors": "ignore",
            "sample.fraction": 0.8625101083119422,
            "splitrule": "extratrees",
            "task_id": "41157",
            "trainsize": 0.525,
            "num.random.splits": 100,
        }
    )
    dynamic_prior_scenario = Scenario(dynamic_prior_base_configuration_space, n_trials=20)
    dynamic_prior_acquisition_function = DynamicPriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(dynamic_prior_scenario),
        prior_configspace=dynamic_prior_prior_configuration_space,
        decay_beta=dynamic_prior_scenario.n_trials / 10,
    )
    dynamic_prior_intensifier = HyperparameterOptimizationFacade.get_intensifier(dynamic_prior_scenario, max_config_calls=1)
    dynamic_prior_smac = HyperparameterOptimizationFacade(
        dynamic_prior_scenario,
        evaluator.train,
        acquisition_function=dynamic_prior_acquisition_function,
        acquisition_maximizer=LocalAndPriorSearch(
            configspace=dynamic_prior_base_configuration_space,
            acquisition_function=dynamic_prior_acquisition_function,
        ),
        callbacks=[AbstractDynamicPriorCallback({10: dynamic_prior_prior_configuration_space})],
        intensifier=dynamic_prior_intensifier,
        overwrite=True,
    )
    dynamic_prior_smac.optimize()

    plot_trajectory(
        {
            "base_smac": base_smac,
            "prior_smac": prior_smac,
            "dynamic_prior_smac": dynamic_prior_smac,
        }
    )
