import pandas as pd

from dynabo.data_processing.download_all_files import (
    GP_PD1_BASELINE_INCUMBENT_PATH,
    GP_PD1_BASELINE_TABLE_PATH,
    RF_PD1_BASELINE_INCUMBENT_PATH,
    RF_PD1_BASELINE_TABLE_PATH,
    RF_PD1_PRIOR_INCUMBENT_PATH,
    RF_PD1_PRIOR_PRIORS_PATH,
    RF_PD1_PRIOR_TABLE_PATH,
    GP_PD1_PRIOR_INCUMBENT_PATH,
    GP_PD1_PRIOR_TABLE_PATH,
    GP_PD1_PRIOR_PRIORS_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH,
    RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH,
    RF_PD1_BASELINE_DECEIVING_LONGER_PATH,
    RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH,
    GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH,
    GP_PD1_BASELINE_DECEIVING_LONGER_PATH,
    GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH,
    RF_PD1_DYNAMIC_PRIORS_TABLE_PATH,
    RF_PD1_DYNAMIC_PRIORS_INCUMBENT_PATH,
    RF_PD1_DYNAMIC_PRIORS_PRIORS_PATH,
    PRIOR_DECAY_ABLATION_TABLE_PATH,
    PRIOR_DECAY_ABLATION_INCUMBENT_PATH,
    PRIOR_DECAY_ABLATION_PRIOR_PATH,
    REMOVE_OLD_PRIORS_ABLATION_TABLE_PATH,
    REMOVE_OLD_PRIORS_ABLATION_INCUMBENT_PATH,
    REMOVE_OLD_PRIORS_ABLATION_PRIOR_PATH,
    MIXED_PRIORS_TABLE_PATH,
    MIXED_PRIORS_INCUMBENT_PATH,
    MIXED_PRIORS_PRIORS_PATH,
    LCB_PD1_BASELINE_TABLE_PATH,
    LCB_PD1_BASELINE_INCUMBENT_PATH,
    LCB_PD1_PRIOR_TABLE_PATH,
    LCB_PD1_PRIOR_INCUMBENT_PATH,
    LCB_PD1_PRIOR_PRIORS_PATH,
    PRIOR_COMBINATION_ABLATION_TABLE_PATH,
    PRIOR_COMBINATION_ABLATION_INCUMBENT_PATH,
    PRIOR_COMBINATION_ABLATION_PRIOR_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_regret,
    create_mixed_plot,
    create_overall_plot,
    create_overall_plot_longer,
    create_scenario_plots,
    create_all_scenarios_plot,
    filter_prior_approach,
    get_min_costs,
    merge_df,
    create_deceiving_longer_scenarios,
    create_final_cost_boxplot_rejection,
    plot_final_run,
    set_ax_style,
)
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def load_cost_data_mfpbench(surrogate: str):
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    if surrogate not in ["rf", "gp"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose either 'rf' or 'gp'.")

    if surrogate == "rf":
        baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
        baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(RF_PD1_PRIOR_TABLE_PATH)
        prior_configs = pd.read_csv(RF_PD1_PRIOR_INCUMBENT_PATH)
        prior_priors = pd.read_csv(RF_PD1_PRIOR_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)
    else:
        baseline_table = pd.read_csv(GP_PD1_BASELINE_TABLE_PATH)
        baseline_config_df = pd.read_csv(GP_PD1_BASELINE_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(GP_PD1_PRIOR_TABLE_PATH)
        prior_configs = pd.read_csv(GP_PD1_PRIOR_INCUMBENT_PATH)
        prior_priors = pd.read_csv(GP_PD1_PRIOR_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_misleading_longer_results_mfpbench(surrogate: str):
    def load_cost_data_misleading_longer_mfpbench():
        """
        Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
        """
        if surrogate == "rf":
            baseline_table = pd.read_csv(RF_PD1_BASELINE_DECEIVING_LONGER_PATH)
            baseline_config_df = pd.read_csv(RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH)
            baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

            prior_table = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH)
            prior_configs = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH)
            prior_priors = pd.read_csv(RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH)
            prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

            min_costs = get_min_costs(benchmarklib="mfpbench")
            baseline_config_df, _ = add_regret([baseline_config_df, prior_config_df], min_costs, benchmarklib="mfpbench")
            prior_config_df, prior_priors_df = add_regret([prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")
        elif surrogate == "gp":
            baseline_table = pd.read_csv(GP_PD1_BASELINE_DECEIVING_LONGER_PATH)
            baseline_config_df = pd.read_csv(GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH)
            baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

            prior_table = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH)
            prior_configs = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH)
            prior_priors = pd.read_csv(GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH)
            prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

            min_costs = get_min_costs(benchmarklib="mfpbench")
            baseline_config_df, _ = add_regret([baseline_config_df, prior_config_df], min_costs, benchmarklib="mfpbench")
            prior_config_df, prior_priors_df = add_regret([prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

        return baseline_config_df, prior_config_df, prior_priors_df

    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_misleading_longer_mfpbench()

    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
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
    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
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
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=None,
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        "DynaBO - accept": accept_all_priors_configs,
        r"$\pi$BO": pibo_incumbent_df,
        "DynaBO - validation": threshold_incumbent_df,
    }
    prior_dict = {
        "DynaBO, accept all priors": accept_all_priors_priors,
        r"$\pi$BO": pibo_prior_df,
        "DynaBO": threshold_prior_df,
    }

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": None, "linestyle": (0, ())},  # Black, solid
        "DynaBO - accept": {"color": "#E69F00", "marker": None, "linestyle": (0, (1, 1))},  # Sky Blue, densely dotted
        r"$\pi$BO": {"color": "#009E73", "marker": None, "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO - validation": {"color": "#D55E00", "marker": None, "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_deceiving_longer_scenarios(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=prior_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/deceiving_longer_results/{surrogate}/regret",
        ncol=4,
    )
    create_overall_plot_longer(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/deceiving_longer_results/{surrogate}/regret", ncol=4)

    # Cost plots
    create_deceiving_longer_scenarios(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=prior_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/deceiving_longer_results/{surrogate}/cost",
        ncol=4,
        y_column="cost",
    )
    create_overall_plot_longer(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/deceiving_longer_results/{surrogate}/cost", ncol=4, y_column="cost")


def plot_prior_rejection_ablation(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench(surrogate=surrogate)

    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
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
    )

    thresholds = [-1, -0.5, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.25, 0.5, 1]
    config_dict = {"Vanilla BO": baseline_config_df, "DynaBO, accept all priors": accept_all_priors_configs}
    prior_dict = {"DynaBO, accept all priors": accept_all_priors_priors}
    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "DynaBO, accept all priors": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
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
            prior_decay_enumerator=5,
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
        base_path=f"plots/prior_rejection_ablation/{surrogate}",
        ncol=int(len(style_dict) / 2) + 1,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/prior_rejection_ablation/{surrogate}", ncol=int(len(style_dict) / 2) + 1)


def plot_prior_rejection_ablation_barplot(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench(surrogate=surrogate)

    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
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

    thresholds = [0.5, 0.25, 0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.5, -1]
    colors_palette = sns.color_palette("colorblind")[:]
    config_dict = {r"$\infty$": baseline_config_df}
    style_dict = {
        r"$\infty$": colors_palette[0],
    }
    colors_palette = colors_palette[2:]
    # Standard seaborn color palette
    for entry, threshold in enumerate(thresholds):
        threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
            incumbent_df=prior_config_df,
            prior_df=prior_prior_df,
            select_dynabo=True,
            select_pibo=False,
            prior_decay_enumerator=5,
            prior_std_denominator=5,
            prior_static_position=True,
            n_prior_based_samples=0,
            prior_every_n_trials=10,
            validate_prior=True,
            prior_validation_method="difference",
            prior_validation_manwhitney_p=None,
            prior_validation_difference_threshold=threshold,
            remove_old_priors=False,
        )
        config_dict[r"$\tau$=" + str(threshold)] = threshold_incumbent_df
        style_dict[r"$\tau$=" + str(threshold)] = colors_palette[entry % len(colors_palette)]

    config_dict[r"$-\infty$"] = accept_all_priors_configs
    style_dict[r"$-\infty$"] = colors_palette[1]

    create_final_cost_boxplot_rejection(config_dict, style_dict, benchmarklib="mfpbench", base_path=f"plots/prior_rejection_ablation/{surrogate}/final_cost_barplot")


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
            prior_decay_enumerator=5,
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


def plot_dynamic_prior_location(surrogate: str):
    def load_dynamic_cost_data(surrogate: str):
        baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
        baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(RF_PD1_DYNAMIC_PRIORS_TABLE_PATH)
        prior_configs = pd.read_csv(RF_PD1_DYNAMIC_PRIORS_INCUMBENT_PATH)
        prior_priors = pd.read_csv(RF_PD1_DYNAMIC_PRIORS_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

        min_costs = get_min_costs(benchmarklib="mfpbench")
        baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

        return baseline_config_df, prior_config_df, prior_priors_df

    baseline_config_df, prior_config_df, prior_prior_df = load_dynamic_cost_data(surrogate=surrogate)

    pibo_all_configs, pibo_all_priors = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=False,
        n_prior_based_samples=None,
        prior_every_n_trials=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=None,
    )

    dynamic_df, dynamic_priors_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=False,
        n_prior_based_samples=0,
        prior_every_n_trials=10,
        validate_prior=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
        prior_chance_theta=0.015,
        remove_old_priors=False,
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        r"$\pi$BO": pibo_all_configs,
        "DynaBO": dynamic_df,
    }
    prior_dict = {}

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        r"$\pi$BO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/dynamic_prior_location/{surrogate}",
        ncol=2,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/dynamic_prior_location/{surrogate}", ncol=len(style_dict))

    # Cost plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/dynamic_prior_location/{surrogate}",
        ncol=2,
        y_column="cost",
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/dynamic_prior_location/{surrogate}", ncol=len(style_dict), y_column="cost")


def plot_final_results_mfpbench(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench(surrogate=surrogate)
    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
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
    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
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
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
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
        # "DynaBO, accept all priors": accept_all_priors_configs,
        r"$\pi$BO": pibo_incumbent_df,
        "DynaBO": threshold_incumbent_df,
    }
    prior_dict = {
        # "DynaBO, accept all priors": accept_all_priors_priors,
        r"$\pi$BO": pibo_prior_df,
        "DynaBO": threshold_prior_df,
    }

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "DynaBO, accept all priors": {"color": "#E69F00", "marker": "s", "linestyle": (0, (1, 1))},  # Sky Blue, densely dotted
        "DynaBO, accept all priors (removed)": {"color": "#2EA9A7", "marker": "s", "linestyle": (0, (3, 5, 1, 5))},  # Sky Blue, densely dotted
        r"$\pi$BO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
        "DynaBO (removed)": {"color": "#CC79A7", "marker": "v", "linestyle": (0, (3, 5, 1, 5))},  # Pink, dash-dot dense
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/final_result_plots/{surrogate}", ncol=len(style_dict))
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
    )

    # Cost plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
        y_column="cost",
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=f"plots/final_result_plots/{surrogate}", ncol=len(style_dict), y_column="cost")
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
        y_column="cost",
    )


def load_prior_decay_ablation(surrogate: str):
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    if surrogate not in ["rf"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose either 'rf' or 'gp'.")

    baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(PRIOR_DECAY_ABLATION_TABLE_PATH)
    prior_configs = pd.read_csv(PRIOR_DECAY_ABLATION_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PRIOR_DECAY_ABLATION_PRIOR_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_decay_ablation(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_prior_decay_ablation(surrogate=surrogate)
    logratithmic_decay_configs, logratithmic_decay_priors = filter_prior_approach(
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
        prior_decay="logarithmic",
    )

    linear_decay_configs, linear_decay_priors = filter_prior_approach(
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
        prior_decay="linear",
    )
    quadratic_decay_configs, quadratic_decay_priors = filter_prior_approach(
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
        prior_decay="quadratic",
    )

    cubic_decay_configs, cubic_decay_priors = filter_prior_approach(
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
        prior_decay="cubic",
    )

    to_four_decay_configs, to_four_decay_priors = filter_prior_approach(
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
        prior_decay="^4",
    )

    to_five_decay_configs, to_five_decay_priors = filter_prior_approach(
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
        prior_decay="^5",
    )

    # Compute Average regret for each scneario and each decay method
    config_dict = {
        "Logarithmic Decay": logratithmic_decay_configs,
        "Linear Decay": linear_decay_configs,
        "Quadratic Decay": quadratic_decay_configs,
        "Cubic Decay": cubic_decay_configs,
        "To the Power of 4 Decay": to_four_decay_configs,
        "To the Power of 5 Decay": to_five_decay_configs,
    }

    prior_types = ["good", "medium", "misleading", "deceiving"]

    rows = []

    for key, df in config_dict.items():
        row = {"config": key}
        for p in prior_types:
            subset = df[df["prior_kind"] == p]["final_regret"]

            mean_val = subset.mean()
            std_val = subset.std(ddof=1)  # sample std
            n = subset.shape[0]
            stderr_val = std_val / np.sqrt(n) if n > 0 else np.nan

            # round both mean and stderr to 3 decimals
            row[f"{p}_mean"] = round(mean_val, 3) if not np.isnan(mean_val) else np.nan
            row[f"{p}_stderr"] = round(stderr_val, 3) if not np.isnan(stderr_val) else np.nan

        rows.append(row)

    result_df = pd.DataFrame(rows)
    print(result_df)


def load_remove_old_priors_ablation(surrogate):
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    if surrogate not in ["rf"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose either 'rf' or 'gp'.")

    baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(REMOVE_OLD_PRIORS_ABLATION_TABLE_PATH)
    prior_configs = pd.read_csv(REMOVE_OLD_PRIORS_ABLATION_INCUMBENT_PATH)
    prior_priors = pd.read_csv(REMOVE_OLD_PRIORS_ABLATION_PRIOR_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def remove_old_priros_ablation():
    baseline_config_df, prior_config_df, prior_prior_df = load_remove_old_priors_ablation(surrogate="rf")

    dynabo_configs, dynabo_priros = filter_prior_approach(
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
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=False,
        prior_decay="linear",
    )

    remove_old_priors_configs, remove_old_priors_priors = filter_prior_approach(
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
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=True,
        prior_decay="linear",
    )

    # Compute Average regret for each scneario and each decay method
    config_dict = {"Vanilla BO": baseline_config_df, "DynaBO": dynabo_configs, "Remove Old Priors": remove_old_priors_configs}

    prior_dict = {"DynaBO": dynabo_priros}

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "Remove Old Priors": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/remove_old_priors/rf",
        ncol=3,
    )


def plot_prior_combination_ablation():
    baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    sum_table = pd.read_csv(RF_PD1_PRIOR_TABLE_PATH)
    sum_configs = pd.read_csv(RF_PD1_PRIOR_INCUMBENT_PATH)
    sum_priors = pd.read_csv(RF_PD1_PRIOR_PRIORS_PATH)
    sum_config_df, sum_priors_df = merge_df(sum_table, sum_configs, sum_priors)

    product_table = pd.read_csv(PRIOR_COMBINATION_ABLATION_TABLE_PATH)
    product_configs = pd.read_csv(PRIOR_COMBINATION_ABLATION_INCUMBENT_PATH)
    product_priors = pd.read_csv(PRIOR_COMBINATION_ABLATION_PRIOR_PATH)
    product_config_df, product_priors_df = merge_df(product_table, product_configs, product_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, sum_config_df, sum_priors_df, product_config_df, product_priors_df = add_regret(
        [baseline_config_df, sum_config_df, sum_priors_df, product_config_df, product_priors_df],
        min_costs,
        benchmarklib="mfpbench",
    )

    sum_dynabo_configs, sum_dynabo_priors = filter_prior_approach(
        incumbent_df=sum_config_df,
        prior_df=sum_priors_df,
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
        prior_decay="linear",
    )

    product_dynabo_configs, product_dynabo_priors = filter_prior_approach(
        incumbent_df=product_config_df,
        prior_df=product_priors_df,
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
        prior_decay="linear",
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        "DynaBO (sum)": sum_dynabo_configs,
        "DynaBO (product)": product_dynabo_configs,
    }
    prior_dict = {
        "DynaBO (sum)": sum_dynabo_priors,
        "DynaBO (product)": product_dynabo_priors,
    }
    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},
        "DynaBO (sum)": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},
        "DynaBO (product)": {"color": "#0072B2", "marker": "s", "linestyle": (0, (3, 5, 1, 5))},
    }

    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/prior_combination_ablation/rf",
        ncol=3,
    )
    create_overall_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        benchmarklib="mfpbench",
        base_path="plots/prior_combination_ablation/rf",
        ncol=3,
    )

    # Standalone legend figure
    handles = [plt.Line2D([0], [0], color=style_dict[k]["color"], linestyle=style_dict[k]["linestyle"], label=k) for k in style_dict]
    fig_legend, ax_legend = plt.subplots(figsize=(6, 0.4))
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, loc="center", ncol=len(style_dict), frameon=False, fontsize=10)
    os.makedirs("plots/prior_combination_ablation/rf", exist_ok=True)
    fig_legend.savefig("plots/prior_combination_ablation/rf/legend.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_legend)


def load_mixed_priors(surrogate):
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    if surrogate not in ["rf"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose either 'rf' or 'gp'.")

    baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(MIXED_PRIORS_TABLE_PATH)
    prior_configs = pd.read_csv(MIXED_PRIORS_INCUMBENT_PATH)
    prior_priors = pd.read_csv(MIXED_PRIORS_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_mixed_priors():
    baseline_config_df, prior_config_df, prior_prior_df = load_mixed_priors(surrogate="rf")

    dynabo_configs, dynabo_priros = filter_prior_approach(
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
        prior_decay="linear",
    )

    remove_old_priors_configs, remove_old_priors_priors = filter_prior_approach(
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
        remove_old_priors=True,
        prior_decay="linear",
    )

    # Compute Average regret for each scneario and each decay method
    config_dict = {"DynaBO": dynabo_configs, "Remove Old Priors": remove_old_priors_configs}

    prior_dict = {"DynaBO": dynabo_priros}

    style_dict = {
        "Remove Old Priors": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_mixed_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/mixed_priors/rf",
        ncol=3,
    )


def load_cost_data_mfpbench_lcb():
    """
    Load the cost data for pd1 LCB experiments, saved in the filesystem.
    """
    baseline_table = pd.read_csv(LCB_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(LCB_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(LCB_PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(LCB_PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(LCB_PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_mfpbench_lcb():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench_lcb()
    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
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
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
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
        r"$\pi$BO": pibo_incumbent_df,
        "DynaBO": threshold_incumbent_df,
    }
    prior_dict = {
        r"$\pi$BO": pibo_prior_df,
        "DynaBO": threshold_prior_df,
    }

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        r"$\pi$BO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},  # Green, dash-dot
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/final_result_plots/lcb",
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path="plots/final_result_plots/lcb", ncol=len(style_dict))
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/final_result_plots/lcb",
        ncol=4,
    )

    # Cost plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/final_result_plots/lcb",
        ncol=4,
        y_column="cost",
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path="plots/final_result_plots/lcb", ncol=len(style_dict), y_column="cost")
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/final_result_plots/lcb",
        ncol=4,
        y_column="cost",
    )


def plot_cover_page_mfpbench(surrogate: str):
    from dynabo.plotting.pc_comparison import load_cost_data_pc

    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench(surrogate=surrogate)
    pc_baseline, pc_dist_prior, pc_point_prior = load_cost_data_pc()

    dynabo_incumbent_df, dynabo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        n_prior_based_samples=0,
        validate_prior=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
        remove_old_priors=False,
    )
    pibo_incumbent_df, _ = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=None,
    )

    prior_kinds = ["good", "deceiving"]

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": None, "linestyle": "-"},
        r"$\pi$BO": {"color": "#009E73", "marker": None, "linestyle": "-."},
        "DynaBO (ours)": {"color": "#D55E00", "marker": None, "linestyle": "--"},
    }
    pcs_color = "#0072B2"
    pcs_linestyle = ":"

    base_path = f"plots/cover_page/{surrogate}"

    import matplotlib.transforms as transforms

    for y_column in ["regret", "cost"]:
        y_label = "Regret" if y_column == "regret" else "Cost"
        os.makedirs(f"{base_path}/{y_column}", exist_ok=True)

        for prior_kind in prior_kinds:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 4), dpi=300)

            config_dict = {
                "Vanilla BO": baseline_config_df,
                r"$\pi$BO": pibo_incumbent_df,
                "DynaBO (ours)": dynabo_incumbent_df,
            }
            ax = plot_final_run(
                config_dict,
                {"DynaBO": dynabo_prior_df},
                style_dict,
                None,
                None,
                prior_kind,
                ax=ax,
                benchmarklib="mfpbench",
                min_ntrials=1,
                max_ntrials=50,
                error_bar_type="se",
                y_column=y_column,
            )

            if y_column == "regret":
                pc_filtered = pc_point_prior[pc_point_prior["prior_kind"] == prior_kind].copy()
                sns.lineplot(
                    x="after_n_evaluations",
                    y="regret",
                    data=pc_filtered,
                    label="i-PC",
                    ax=ax,
                    errorbar="se",
                    color=pcs_color,
                    linestyle=pcs_linestyle,
                    drawstyle="steps-pre",
                )

            set_ax_style(ax, x_label="Number of Evaluations", y_label=y_label, benchmarklib="mfpbench", auto_yticks=(y_column != "regret"), xticks=[0, 10, 20, 30, 40, 50])
            ax.tick_params(axis="both", labelsize=18.5)
            ax.set_xlabel("Number of Evaluations", fontsize=18.5, fontweight="bold")
            if y_column == "regret":
                ax.set_ylim(bottom=0.02, top=0.135)
                ax.set_yticks([0.04, 0.06, 0.08, 0.10, 0.12])

            blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            prior_locs = dynabo_prior_df[dynabo_prior_df["prior_kind"] == prior_kind]["after_n_evaluations"].unique()
            for loc in sorted(prior_locs):
                ax.plot(
                    loc,
                    1.0,
                    marker="v",
                    color="#E69F00",
                    markersize=10,
                    transform=blended,
                    clip_on=False,
                    zorder=5,
                )

            label_order = ["i-PC", "Vanilla BO", r"$\pi$BO", "DynaBO (ours)"] if y_column == "regret" else ["Vanilla BO", r"$\pi$BO", "DynaBO (ours)"]
            label_line_dict = {line.get_label(): line for line in ax.get_lines() if not line.get_label().startswith("_")}
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            for lbl in label_order:
                if lbl not in label_line_dict:
                    continue
                line = label_line_dict[lbl]
                ydata = line.get_ydata()
                if len(ydata) == 0:
                    continue
                valid = ydata[~np.isnan(ydata)]
                if len(valid) == 0:
                    continue
                display_label = lbl.replace(" (ours)", "")
                below = lbl in (r"$\pi$BO", "Vanilla BO")
                if lbl == r"$\pi$BO":
                    y_offset = -0.003
                elif lbl == "Vanilla BO":
                    y_offset = 0.0
                elif lbl == "i-PC":
                    y_offset = 0.003
                else:
                    y_offset = 0.006
                ax.text(
                    0.97,
                    valid[-1] + y_offset,
                    display_label,
                    fontsize=18.5,
                    color=line.get_color(),
                    va="top" if below else "bottom",
                    ha="right",
                    transform=trans,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
                )
            fig.tight_layout()
            plt.savefig(
                f"{base_path}/{y_column}/{prior_kind}.pdf",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()


def plot_cover_page_mfpbench_no_pc(surrogate: str):
    import matplotlib.transforms as transforms

    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench(surrogate=surrogate)

    dynabo_incumbent_df, dynabo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=10,
        n_prior_based_samples=0,
        validate_prior=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.15,
        remove_old_priors=False,
    )
    pibo_incumbent_df, _ = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=5,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=None,
    )

    prior_kinds = ["good", "deceiving"]

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": None, "linestyle": "-"},
        r"$\pi$BO": {"color": "#009E73", "marker": None, "linestyle": "-."},
        "DynaBO (ours)": {"color": "#D55E00", "marker": None, "linestyle": "--"},
    }

    base_path = f"plots/cover_page_no_pc/{surrogate}"

    for y_column in ["regret", "cost"]:
        y_label = "Regret" if y_column == "regret" else "Cost"
        os.makedirs(f"{base_path}/{y_column}", exist_ok=True)

        for prior_kind in prior_kinds:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 4), dpi=300)

            config_dict = {
                "Vanilla BO": baseline_config_df,
                r"$\pi$BO": pibo_incumbent_df,
                "DynaBO (ours)": dynabo_incumbent_df,
            }
            ax = plot_final_run(
                config_dict,
                {"DynaBO": dynabo_prior_df},
                style_dict,
                None,
                None,
                prior_kind,
                ax=ax,
                benchmarklib="mfpbench",
                min_ntrials=1,
                max_ntrials=50,
                error_bar_type="se",
                y_column=y_column,
            )

            set_ax_style(ax, x_label="Number of Evaluations", y_label=y_label, benchmarklib="mfpbench", auto_yticks=(y_column != "regret"), xticks=[0, 10, 20, 30, 40, 50])
            ax.tick_params(axis="both", labelsize=18.5)
            ax.set_xlabel("Number of Evaluations", fontsize=18.5, fontweight="bold")
            if y_column == "regret":
                ax.set_ylim(bottom=0.02, top=0.135)
                ax.set_yticks([0.04, 0.06, 0.08, 0.10, 0.12])

            blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            prior_locs = dynabo_prior_df[dynabo_prior_df["prior_kind"] == prior_kind]["after_n_evaluations"].unique()
            for loc in sorted(prior_locs):
                ax.plot(
                    loc,
                    1.0,
                    marker="v",
                    color="#E69F00",
                    markersize=10,
                    transform=blended,
                    clip_on=False,
                    zorder=5,
                )

            label_order = ["Vanilla BO", r"$\pi$BO", "DynaBO (ours)"]
            label_line_dict = {line.get_label(): line for line in ax.get_lines() if not line.get_label().startswith("_")}
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            for lbl in label_order:
                if lbl not in label_line_dict:
                    continue
                line = label_line_dict[lbl]
                ydata = line.get_ydata()
                if len(ydata) == 0:
                    continue
                valid = ydata[~np.isnan(ydata)]
                if len(valid) == 0:
                    continue
                display_label = lbl.replace(" (ours)", "")
                below = lbl in (r"$\pi$BO", "Vanilla BO")
                if lbl == r"$\pi$BO":
                    y_offset = -0.003
                elif lbl == "Vanilla BO":
                    y_offset = 0.0
                else:
                    y_offset = 0.006
                ax.text(
                    0.97,
                    valid[-1] + y_offset,
                    display_label,
                    fontsize=18.5,
                    color=line.get_color(),
                    va="top" if below else "bottom",
                    ha="right",
                    transform=trans,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=1),
                )
            fig.tight_layout()
            plt.savefig(
                f"{base_path}/{y_column}/{prior_kind}.pdf",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    # plot_dynamic_prior_location("rf")
    # plot_final_results_mfpbench("rf")
    # plot_prior_rejection_ablation_barplot("rf")
    # plot_misleading_longer_results_mfpbench("gp")
    # plot_misleading_longer_results_mfpbench("rf")
    # plot_decay_ablation("rf")
    # remove_old_priros_ablation()
    # plot_mixed_priors()
    # plot_final_results_mfpbench_lcb()
    # plot_prior_combination_ablation()
    # plot_cover_page_mfpbench("rf")
    plot_cover_page_mfpbench_no_pc("rf")
