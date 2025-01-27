from typing import Dict

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, NormalFloatHyperparameter, NormalIntegerHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter


def build_prior_configuration_space(configuration_space: ConfigurationSpace, prior: Dict[str, float], prior_std_denominator: float) -> ConfigurationSpace:
    configspace_name = configuration_space.name

    random_state = configuration_space.random.get_state()
    new_configuration_space = ConfigurationSpace(name=f"{configspace_name}_prior")
    new_configuration_space.random.set_state(random_state)

    for hyperparameter in configuration_space.values():
        if isinstance(hyperparameter, CategoricalHyperparameter):
            old_choices = hyperparameter.choices
            optimum_choice = prior.get(hyperparameter.name, None)
            if not pd.isna(optimum_choice):
                new_weights = [1 if choice == optimum_choice else 0 for choice in old_choices]
            else:
                new_weights = [1 / len(old_choices)] * len(old_choices)
            new_hyperparameter = CategoricalHyperparameter(
                name=hyperparameter.name,
                choices=hyperparameter.choices,
                default_value=hyperparameter.default_value,
                weights=new_weights,
            )
        elif isinstance(hyperparameter, UniformFloatHyperparameter):
            optimum_choice = prior.get(hyperparameter.name, None)
            if not pd.isna(optimum_choice):
                new_hyperparameter = NormalFloatHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    mu=optimum_choice,
                    sigma=(hyperparameter.upper - hyperparameter.lower) / prior_std_denominator,
                    log=hyperparameter.log,
                )
            else:
                new_hyperparameter = UniformFloatHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                )
        elif isinstance(hyperparameter, UniformIntegerHyperparameter):
            # Deactiated Hyperparameter sampled uniform
            optimum_choice = prior.get(hyperparameter.name, None)
            if not pd.isna(optimum_choice):
                new_hyperparameter = NormalIntegerHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    mu=optimum_choice,
                    sigma=(hyperparameter.upper - hyperparameter.lower) / prior_std_denominator,
                    log=hyperparameter.log,
                )
            else:
                new_hyperparameter = UniformIntegerHyperparameter(
                    name=hyperparameter.name,
                    lower=hyperparameter.lower,
                    upper=hyperparameter.upper,
                    log=hyperparameter.log,
                )
        elif isinstance(hyperparameter, Constant):
            new_hyperparameter = hyperparameter
        else:
            raise NotImplementedError(f"Hyperparameter {hyperparameter} not supported.")

        new_configuration_space.add(new_hyperparameter)
    return new_configuration_space
