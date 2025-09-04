import pandas as pd

from dynabo.data_processing.download_all_files import (
    PD1_BASELINE_INCUMBENT_PATH,
    PD1_BASELINE_TABLE_PATH,
    PD1_PRIOR_INCUMBENT_PATH,
    PD1_PRIOR_PRIORS_PATH,
    PD1_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, create_overall_plot, create_scenario_plots, filter_prior_approach, get_min_costs, merge_df
import seaborn as sns


def load_cost_data_mfpbench():
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """
    baseline_table = pd.read_csv(PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs([baseline_config_df, prior_config_df], benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_mfpbench():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench()
    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=50,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=False,
        n_prior_based_samples=0,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )
    # baseline_perfect_incumbent_df, baseline_perfect_prior_df = filter_prior_approach(
    #    incumbent_df=prior_config_df,
    #    prior_df=prior_prior_df,
    #    select_dynabo=True,
    #    select_pibo=False,
    #    prior_decay_enumerator=50,
    #    prior_std_denominator=5,
    #    prior_static_position=True,
    #    prior_every_n_trials=10,
    #    validate_prior=True,
    #    prior_validation_method="baseline_perfect",
    #    prior_validation_manwhitney_p=None,
    #    prior_validation_difference_threshold=None,
    # )
    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=50,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        n_prior_based_samples=0,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
    )
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=50,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        "DynaBO, accept all priors": accept_all_priors_configs,
        r"$\pi$BO": pibo_incumbent_df,
        # "DynaBO, perfect validation": baseline_perfect_incumbent_df,
        "DynaBO, threshold validation": threshold_incumbent_df,
    }
    prior_dict = {
        "DynaBO, accept all priors": accept_all_priors_priors,
        r"$\pi$BO": pibo_prior_df,
        # "DynaBO, perfect validation": baseline_perfect_prior_df,
        "DynaBO, threshold validation": threshold_prior_df,
    }

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "DynaBO, accept all priors": {"color": "#E69F00", "marker": "s", "linestyle": (0, (1, 1))},  # Sky Blue, densely dotted
        "DynaBO, accept all priors (3 samples)": {"color": "#2EA9A7", "marker": "s", "linestyle": (0, (3, 5, 1, 5))},  # Sky Blue, densely dotted
        r"$\pi$BO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        # "DynaBO, perfect validation": {"color": "#F0E442", "marker": "s", "linestyle": (0, (1, 1))},  # Blue, dash-dot-dot
        "DynaBO, threshold validation": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/final_result_plots",
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="mfpbench", base_path="plots/final_result_plots", ncol=len(style_dict))


def plot_prior_rejection_ablation():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench()

    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=50,
        prior_std_denominator=5,
        n_prior_based_samples=0,
        prior_static_position=None,
        prior_every_n_trials=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    thresholds = [-1, -0.5, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.25, 0.5, 1]
    config_dict = {"Vanilla BO": baseline_config_df, "PiBO": pibo_incumbent_df}
    prior_dict = {"PiBO": pibo_prior_df}
    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "PiBO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
    }
    # Standard seaborn color palette
    colors_palette = sns.color_palette("colorblind")[1:]
    # Define a list of unique linestyles for each threshold
    linestyles = [
        (0, (1, 1)),  # dotted
        (0, (3, 1, 1, 1)),  # dash-dot-dot
        (0, (5, 2)),  # dashed
        (0, (3, 5, 1, 5)),  # dash-dot
        (0, (1, 10)),  # sparse dots
        (0, (5, 10)),  # sparse dashes
        (0, (3, 1, 1, 1, 1, 1)),  # custom
        (0, (2, 2)),  # short dash
        (0, (4, 1, 1, 1)),  # dash-dot
        (0, (1, 5)),  # sparse dashes
        (0, (3, 5)),  # custom
    ]
    for entry, threshold in enumerate(thresholds):
        threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
            incumbent_df=prior_config_df,
            prior_df=prior_prior_df,
            select_dynabo=True,
            select_pibo=False,
            prior_decay_enumerator=50,
            prior_std_denominator=5,
            prior_static_position=True,
            n_prior_based_samples=0,
            prior_every_n_trials=10,
            validate_prior=True,
            prior_validation_method="difference",
            prior_validation_manwhitney_p=None,
            prior_validation_difference_threshold=threshold,
        )
        config_dict[r"$\tau$=" + str(threshold)] = threshold_incumbent_df
        prior_dict[r"$\tau$=" + str(threshold)] = threshold_prior_df
        style_dict[r"$\tau$=" + str(threshold)] = {"color": colors_palette[entry % len(colors_palette)], "marker": "v", "linestyle": linestyles[entry % len(linestyles)]}

    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/prior_rejection_ablation",
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="mfpbench", base_path="plots/prior_rejection_ablation", ncol=len(style_dict))


def plot_prior_based_samples_ablation():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench()

    n_prior_based_samples = [0, 1, 2, 3, 4]
    config_dict = {"Vanilla BO": baseline_config_df}
    prior_dict = {}
    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "PiBO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
    }
    # Standard seaborn color palette
    colors_palette = sns.color_palette("colorblind")[1:]
    # Define a list of unique linestyles for each threshold
    linestyles = [
        (0, (1, 1)),  # dotted
        (0, (3, 1, 1, 1)),  # dash-dot-dot
        (0, (5, 2)),  # dashed
        (0, (3, 5, 1, 5)),  # dash-dot
        (0, (1, 10)),  # sparse dots
        (0, (5, 10)),  # sparse dashes
        (0, (3, 1, 1, 1, 1, 1)),  # custom
        (0, (2, 2)),  # short dash
        (0, (4, 1, 1, 1)),  # dash-dot
    ]
    for entry, n_prior_based_samples in enumerate(n_prior_based_samples):
        threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
            incumbent_df=prior_config_df,
            prior_df=prior_prior_df,
            select_dynabo=True,
            select_pibo=False,
            prior_decay_enumerator=50,
            prior_std_denominator=5,
            prior_static_position=True,
            n_prior_based_samples=n_prior_based_samples,
            prior_every_n_trials=10,
            validate_prior=True,
            prior_validation_method="difference",
            prior_validation_manwhitney_p=None,
            prior_validation_difference_threshold=-0.25,
        )
        config_dict[f"{entry} samples"] = threshold_incumbent_df
        prior_dict[f"{entry} samples"] = threshold_prior_df
        style_dict[f"{entry} samples"] = {"color": colors_palette[entry % len(colors_palette)], "marker": "v", "linestyle": linestyles[entry % len(linestyles)]}

    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/prior_based_samples_ablation",
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="mfpbench", base_path="plots/prior_based_samples_ablation", ncol=len(style_dict))


if __name__ == "__main__":
    plot_final_results_mfpbench()
    plot_prior_rejection_ablation()
    plot_prior_based_samples_ablation()
