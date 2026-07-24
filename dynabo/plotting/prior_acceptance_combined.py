import pandas as pd

from dynabo.data_processing.download_all_files import (
    RF_PD1_PRIOR_INCUMBENT_PATH,
    RF_PD1_PRIOR_PRIORS_PATH,
    RF_PD1_PRIOR_TABLE_PATH,
    YAHPO_PRIOR_INCUMBENT_PATH,
    YAHPO_PRIOR_PRIORS_PATH,
    YAHPO_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_prior_position,
    create_prior_acceptance_grid,
    filter_prior_approach,
    merge_df,
)

# Column order of the joined grid (4 PD1/mfpbench scenarios + 2 yahpogym scenarios).
SCENARIO_ORDER = [
    "cifar100_wideresnet_2048",
    "imagenet_resnet_512",
    "lm1b_transformer_2048",
    "translatewmt_xformer_64",
    "rbv2_xgboost",
    "lcbench",
]
SCENARIO_TITLES = {
    "cifar100_wideresnet_2048": "CIFAR-100",
    "imagenet_resnet_512": "ImageNet",
    "lm1b_transformer_2048": "LM1B",
    "translatewmt_xformer_64": "TranslateWMT",
    "rbv2_xgboost": "rbv2 XGBoost",
    "lcbench": "LCBench",
}

# Columns needed for the grid (dataset differs between benchmarks, so it is dropped after ranking).
KEEP = ["scenario", "prior_kind", "prior_position", "prior_accepted", "superior_configuration", "correct"]


def _load_pd1() -> pd.DataFrame:
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
    prior_df = add_prior_position(prior_df, run_keys=["scenario", "prior_kind", "seed"])
    return prior_df[KEEP]


def _load_yahpo() -> pd.DataFrame:
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
    prior_df = add_prior_position(prior_df, run_keys=["scenario", "dataset", "prior_kind", "seed"])
    return prior_df[KEEP]


def plot_prior_acceptance_combined():
    """
    Joined prior-acceptance grid over all six scenarios (4 PD1 + 2 yahpogym). Rows are the four
    prior kinds, columns the scenarios. Percentages aggregate over seeds (and over datasets for the
    yahpogym scenarios). Uses the main-results DynaBO validation configuration (threshold -0.15).
    """
    combined = pd.concat([_load_pd1(), _load_yahpo()], ignore_index=True)
    create_prior_acceptance_grid(
        combined,
        run_keys=None,  # prior_position already computed per benchmark
        base_path="plots/prior_acceptance/combined",
        scenarios=SCENARIO_ORDER,
        scenario_titles=SCENARIO_TITLES,
    )


if __name__ == "__main__":
    plot_prior_acceptance_combined()
