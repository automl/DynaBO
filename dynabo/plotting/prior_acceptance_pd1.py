import pandas as pd

from dynabo.data_processing.download_all_files import (
    RF_PD1_PRIOR_INCUMBENT_PATH,
    RF_PD1_PRIOR_PRIORS_PATH,
    RF_PD1_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    create_prior_acceptance_grid,
    filter_prior_approach,
    merge_df,
)

SCENARIO_TITLES = {
    "cifar100_wideresnet_2048": "CIFAR-100",
    "imagenet_resnet_512": "ImageNet",
    "lm1b_transformer_2048": "LM1B",
    "translatewmt_xformer_64": "TranslateWMT",
}


def plot_prior_acceptance_pd1(surrogate: str = "rf"):
    """
    Prior acceptance overview for the PD1 (mfpbench) main results. One subplot per
    (scenario, prior kind); shows accepted / should-accept / correct-decision rates
    across the four successively provided priors. Uses the main-results DynaBO
    validation configuration (difference threshold -0.15).
    """
    if surrogate != "rf":
        raise ValueError("Only the RF surrogate is available for the PD1 main results.")

    prior_table = pd.read_csv(RF_PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(RF_PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(RF_PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    _, prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_priors_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        n_prior_based_samples=0,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
        remove_old_priors=False,
    )

    create_prior_acceptance_grid(
        prior_df,
        run_keys=["scenario", "prior_kind", "seed"],
        base_path=f"plots/prior_acceptance/pd1/{surrogate}",
        scenario_titles=SCENARIO_TITLES,
    )


if __name__ == "__main__":
    plot_prior_acceptance_pd1("rf")
