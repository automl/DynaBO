import os

import matplotlib.pyplot as plt
import pandas as pd

from dynabo.data_processing.download_all_files import (
    GP_PD1_BASELINE_DECEIVING_LONGER_PATH,
    GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH,
    RF_PD1_BASELINE_DECEIVING_LONGER_PATH,
    RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_regret,
    create_deceiving_longer_scenarios,
    create_overall_plot_longer,
    filter_prior_approach,
    get_min_costs,
    merge_df,
)


def load_data(surrogate: str):
    if surrogate == "rf":
        baseline_table = pd.read_csv(RF_PD1_BASELINE_DECEIVING_LONGER_PATH)
        baseline_config_df = pd.read_csv(RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH)
        prior_configs = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH)
        prior_priors = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)
    elif surrogate == "gp":
        baseline_table = pd.read_csv(GP_PD1_BASELINE_DECEIVING_LONGER_PATH)
        baseline_config_df = pd.read_csv(GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH)
        prior_configs = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH)
        prior_priors = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)
    else:
        raise ValueError(f"Unknown surrogate: {surrogate}")

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, _ = add_regret([baseline_config_df, prior_config_df], min_costs, benchmarklib="mfpbench")
    prior_config_df, prior_priors_df = add_regret([prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def save_legend(style_dict: dict, path: str):
    handles = [
        plt.Line2D([0], [0], color=v["color"], linestyle=v["linestyle"], label=k)
        for k, v in style_dict.items()
    ]
    fig, ax = plt.subplots(figsize=(len(style_dict) * 2, 0.4))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(style_dict), frameon=False, fontsize=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rejection_comparison(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_data(surrogate)

    accept_all_configs, accept_all_priors = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=False,
        n_prior_based_samples=0,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=False,
    )
    rejection_configs, rejection_priors = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
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
    base_style = {
        "Vanilla BO": {"color": "#000000", "marker": None, "linestyle": (0, ())},
        "DynaBO - accept": {"color": "#E69F00", "marker": None, "linestyle": (0, (1, 1))},
        "DynaBO - validation": {"color": "#D55E00", "marker": None, "linestyle": (0, (1, 1))},
    }

    # Plot 1: all variants
    config_dict_all = {
        "Vanilla BO": baseline_config_df,
        "DynaBO - accept": accept_all_configs,
        "DynaBO - validation": rejection_configs,
    }
    prior_dict_all = {
        "DynaBO - accept": accept_all_priors,
        "DynaBO - validation": rejection_priors,
    }

    base_path_all = f"plots/rejection_comparison/{surrogate}/all_variants"
    create_deceiving_longer_scenarios(
        config_dict_all, prior_dict_all, base_style,
        error_bar_type="se",
        scenarios=prior_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=base_path_all,
        ncol=4,
        figsize=(7, 3.5),
        show_legend=True,
        fmt="png",
    )
    create_overall_plot_longer(
        config_dict_all, prior_dict_all, base_style,
        error_bar_type="se",
        benchmarklib="mfpbench",
        base_path=base_path_all,
        ncol=4,
        figsize=(7, 3.5),
        show_legend=True,
        fmt="png",
    )
    save_legend(base_style, f"{base_path_all}/legend.png")

    # Plot 2: only DynaBO without rejection
    no_rejection_style = {
        "Vanilla BO": {"color": "#000000", "marker": None, "linestyle": (0, ())},
        "DynaBO - accept": {"color": "#E69F00", "marker": None, "linestyle": (0, (1, 1))},
    }
    config_dict_no_rejection = {
        "Vanilla BO": baseline_config_df,
        "DynaBO - accept": accept_all_configs,
    }
    prior_dict_no_rejection = {
        "DynaBO - accept": accept_all_priors,
    }

    base_path_no_rejection = f"plots/rejection_comparison/{surrogate}/no_rejection"
    create_deceiving_longer_scenarios(
        config_dict_no_rejection, prior_dict_no_rejection, no_rejection_style,
        error_bar_type="se",
        scenarios=prior_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=base_path_no_rejection,
        ncol=2,
        figsize=(7, 3.5),
        show_legend=True,
        fmt="png",
    )
    create_overall_plot_longer(
        config_dict_no_rejection, prior_dict_no_rejection, no_rejection_style,
        error_bar_type="se",
        benchmarklib="mfpbench",
        base_path=base_path_no_rejection,
        ncol=2,
        figsize=(7, 3.5),
        show_legend=True,
        fmt="png",
    )
    save_legend(no_rejection_style, f"{base_path_no_rejection}/legend.png")


if __name__ == "__main__":
    # plot_rejection_comparison("gp")
    plot_rejection_comparison("rf")
