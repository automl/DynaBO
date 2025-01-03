from smac.callback import Callback
from typing import Dict
from ConfigSpace import ConfigurationSpace
from smac.main.smbo import SMBO
from dynabo.smac_additions.local_and_prior_search import LocalAndPriorSearch
from dynabo.smac_additions.dynmaic_prior_activation_function import (
    DynamicPriorAcquisitionFunction,
)


class ChangePriorCallback(Callback):
    def __init__(self, intervention_schedule: Dict[int, ConfigurationSpace]):
        super().__init__()

        self.intervention_schedule = intervention_schedule

    def on_iteration_start(self, smbo: SMBO):
        "We add prior information, before the next iteration is started."

        if smbo.runhistory.finished in self.intervention_schedule:
            self.set_prior(smbo)
        return super().on_iteration_start(smbo)

    def set_prior(self, smbo: SMBO):
        new_prior_configspace = self.intervention_schedule[len(smbo.runhistory)] 

        acquisition_function_maximizer: LocalAndPriorSearch = (
            smbo.intensifier.config_selector._acquisition_maximizer
        )
        acquisition_function_maximizer.dynamic_init(new_prior_configspace)

        acquisition_function: DynamicPriorAcquisitionFunction = (
            smbo.intensifier.config_selector._acquisition_function
        )
        acquisition_function.dynamic_init(
            acquisition_function=acquisition_function._acquisition_function,
            prior_configspace=new_prior_configspace,
            decay_beta=smbo._scenario.n_trials / 10,
        )
