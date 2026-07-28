import pandas as pd

from dynabo.data_processing.download_all_files import (
    YAHPO_BASELINE_TABLE_PATH,
    YAHPO_BASELINE_INCUMBENT_PATH,
    YAHPO_PRIOR_TABLE_PATH,
    YAHPO_PRIOR_INCUMBENT_PATH,
    YAHPO_PRIOR_PRIORS_PATH,
    YAHPO_PRIOR_DECAY_ABLATION_TABLE_PATH,
    YAHPO_PRIOR_DECAY_ABLATION_INCUMBENT_PATH,
    YAHPO_PRIOR_DECAY_ABLATION_PRIOR_PATH,
    LCB_YAHPO_BASELINE_TABLE_PATH,
    LCB_YAHPO_BASELINE_INCUMBENT_PATH,
    LCB_YAHPO_PRIOR_TABLE_PATH,
    LCB_YAHPO_PRIOR_INCUMBENT_PATH,
    LCB_YAHPO_PRIOR_PRIORS_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, create_scenario_plots, create_all_scenarios_plot, filter_prior_approach, get_min_costs, merge_df


def load_prior_decay_ablation_yahpogym(surrogate: str) -> str:
    """
    Load the prior decay ablation data for yahpogym and return a markdown table
    with the final regret per scenario and decay type.
    """

    if surrogate not in ["rf"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose 'rf'.")

    baseline_table = pd.read_csv(YAHPO_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(YAHPO_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(YAHPO_PRIOR_DECAY_ABLATION_TABLE_PATH)
    prior_configs = pd.read_csv(YAHPO_PRIOR_DECAY_ABLATION_INCUMBENT_PATH)
    prior_priors = pd.read_csv(YAHPO_PRIOR_DECAY_ABLATION_PRIOR_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    baseline_config_df = baseline_config_df[baseline_config_df["benchmarklib"] == "yahpogym"]
    prior_config_df = prior_config_df[prior_config_df["benchmarklib"] == "yahpogym"]
    prior_priors_df = prior_priors_df[prior_priors_df["benchmarklib"] == "yahpogym"]

    min_costs = get_min_costs(benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="yahpogym")

    prior_kind_titles = {"good": "Expert", "medium": "Advanced", "misleading": "Misleading", "deceiving": "Deceiving"}

    def fmt(agg: pd.DataFrame) -> pd.Series:
        return agg["mean"].map("{:.4f}".format) + " ± " + agg["sem"].map("{:.4f}".format)

    agg = prior_config_df.groupby(["prior_kind", "prior_decay"])["regret"].agg(["mean", "sem"])
    decay_summary = fmt(agg).unstack("prior_decay")
    numeric_means = agg["mean"].unstack("prior_decay")

    row_order = ["good", "medium", "misleading", "deceiving"]
    col_order = ["logarithmic", "linear", "quadratic", "cubic", "^4", "^5"]
    row_order = [r for r in row_order if r in decay_summary.index]
    col_order = [c for c in col_order if c in decay_summary.columns]

    decay_summary = decay_summary.loc[row_order, col_order]
    numeric_means = numeric_means.loc[row_order, col_order]

    decay_summary.index = decay_summary.index.map(prior_kind_titles)
    numeric_means.index = numeric_means.index.map(prior_kind_titles)

    for row in decay_summary.index:
        min_col = numeric_means.loc[row].idxmin()
        decay_summary.loc[row, min_col] = f"**{decay_summary.loc[row, min_col]}**"

    print(decay_summary.to_markdown())


def load_cost_data_yahpogym(surrogate: str):
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    if surrogate not in ["rf", "gp"]:
        raise ValueError(f"Surrogate {surrogate} not recognized. Choose either 'rf' or 'gp'.")

    if surrogate == "rf":
        baseline_table = pd.read_csv(YAHPO_BASELINE_TABLE_PATH)
        baseline_config_df = pd.read_csv(YAHPO_BASELINE_INCUMBENT_PATH)
        baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

        prior_table = pd.read_csv(YAHPO_PRIOR_TABLE_PATH)
        prior_configs = pd.read_csv(YAHPO_PRIOR_INCUMBENT_PATH)
        prior_priors = pd.read_csv(YAHPO_PRIOR_PRIORS_PATH)
        prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

        # If scenario = #lcbench divide cost by 100
        baseline_config_df = baseline_config_df[baseline_config_df["benchmarklib"] == "yahpogym"]
        prior_config_df = prior_config_df[prior_config_df["benchmarklib"] == "yahpogym"]
        prior_priors_df = prior_priors_df[prior_priors_df["benchmarklib"] == "yahpogym"]

    else:
        raise ValueError("GP nto used for Yahpo.")

    min_costs = get_min_costs(benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="yahpogym")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_yahpogym(surrogate: str):
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_yahpogym(surrogate=surrogate)

    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
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
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=20,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=False,
        prior_decay="linear",
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        # "DynaBO, accept all priors": accept_all_priors_configs,
        r"$\pi$BO": pibo_incumbent_df,
        # "DynaBO, perfect validation": baseline_perfect_incumbent_df,
        "DynaBO, threshold validation": threshold_incumbent_df,
    }
    prior_dict = {
        "DynaBO": threshold_prior_df,
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
        benchmarklib="yahpogym",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
    )
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
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
        benchmarklib="yahpogym",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
        y_column="cost",
    )
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
        y_column="cost",
    )
    # create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="yahpogym", base_path=f"plots/final_result_plots/{surrogate}", ncol=len(style_dict))


def load_cost_data_yahpogym_lcb():
    """
    Load the cost data for yahpogym LCB experiments, saved in the filesystem.
    """
    baseline_table = pd.read_csv(LCB_YAHPO_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(LCB_YAHPO_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(LCB_YAHPO_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(LCB_YAHPO_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(LCB_YAHPO_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    baseline_config_df = baseline_config_df[baseline_config_df["benchmarklib"] == "yahpogym"]
    prior_config_df = prior_config_df[prior_config_df["benchmarklib"] == "yahpogym"]
    prior_priors_df = prior_priors_df[prior_priors_df["benchmarklib"] == "yahpogym"]

    baseline_config_df.loc[baseline_config_df["scenario"] == "lcbench", "cost"] /= 100
    prior_config_df.loc[prior_config_df["scenario"] == "lcbench", "cost"] /= 100

    min_costs = get_min_costs(benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="yahpogym")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_yahpogym_lcb():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_yahpogym_lcb()

    threshold_incumbent_df, threshold_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
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
    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=20,
        prior_std_denominator=5,
        prior_static_position=None,
        prior_every_n_trials=None,
        n_prior_based_samples=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
        remove_old_priors=False,
        prior_decay="linear",
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
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},
        r"$\pi$BO": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},
    }
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
        base_path="plots/final_result_plots/lcb/yahpogym",
        ncol=4,
    )
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
        base_path="plots/final_result_plots/lcb/yahpogym",
        ncol=4,
    )
    # Cost plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
        base_path="plots/final_result_plots/lcb/yahpogym",
        ncol=4,
        y_column="cost",
    )
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="yahpogym",
        base_path="plots/final_result_plots/lcb/yahpogym",
        ncol=4,
        y_column="cost",
    )


if __name__ == "__main__":
    # plot_final_results_yahpogym("rf")
    # plot_final_results_yahpogym_lcb()
    load_prior_decay_ablation_yahpogym("rf")
    # plot_final_results_yahpogym_lcb()
