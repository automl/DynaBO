import pandas as pd

from dynabo.data_processing.download_all_files import (
    YAHPO_PRIOR_INCUMBENT_PATH,
    YAHPO_PRIOR_PRIORS_PATH,
    YAHPO_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    create_prior_acceptance_grid,
    filter_prior_approach,
    merge_df,
)

SCENARIO_TITLES = {
    "lcbench": "LCBench",
    "rbv2_xgboost": "rbv2 XGBoost",
}


def plot_prior_acceptance_yahpo(surrogate: str = "rf"):
    """
    Prior acceptance overview for the yahpogym main results. One subplot per
    (scenario, prior kind); shows accepted / should-accept / correct-decision rates
    across the four successively provided priors. Percentages are aggregated over all
    datasets and seeds. Uses the main-results DynaBO validation configuration
    (difference threshold -0.15).
    """
    if surrogate != "rf":
        raise ValueError("Only the RF surrogate is available for the yahpogym main results.")

    prior_table = pd.read_csv(YAHPO_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(YAHPO_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(YAHPO_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    _, prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_priors_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=20,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=40,
        validate_prior=True,
        n_prior_based_samples=0,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
        remove_old_priors=False,
        prior_decay="linear",
    )

    create_prior_acceptance_grid(
        prior_df,
        run_keys=["scenario", "dataset", "prior_kind", "seed"],
        base_path=f"plots/prior_acceptance/yahpo/{surrogate}",
        scenario_titles=SCENARIO_TITLES,
    )


if __name__ == "__main__":
    plot_prior_acceptance_yahpo("rf")
