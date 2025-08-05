# %%

import pandas as pd

from dynabo.data_processing.download_all_files import (
    YAHPO_ABLATION_INCUMBENT_PATH,
    YAHPO_ABLATION_PRIOR_PATH,
    YAHPO_ABLATION_TABLE_PATH,
    YAHPO_BASELINE_INCUMBENT_PATH,
    YAHPO_BASELINE_TABLE_PATH,
    YAHPO_PRIOR_INCUMBENT_PATH,
    YAHPO_PRIOR_PRIORS_PATH,
    YAHPO_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, create_overall_plot, create_scenario_plots, filter_prior_approach, get_best_performances, merge_df


def load_performance_data_yahpo():
    """
    Load the performance data, saved in the filesystem. Do some data_cleaning for lcbench and add regret.
    """

    def _clean_lcbench_performance(df: pd.DataFrame):
        mask = df["scenario"] == "lcbench"
        df.loc[mask, "performance"] = df.loc[mask, "performance"] / 100
        df.loc[mask, "final_performance"] = df.loc[mask, "final_performance"] / 100
        return df

    baseline_table = pd.read_csv(YAHPO_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(YAHPO_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(YAHPO_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(YAHPO_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(YAHPO_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    # For scenario lcbench divide by final_performance and perforamnce by 100
    baseline_config_df = _clean_lcbench_performance(baseline_config_df)
    prior_config_df = _clean_lcbench_performance(prior_config_df)
    prior_priors_df = _clean_lcbench_performance(prior_priors_df)

    best_performances = get_best_performances([baseline_config_df, prior_config_df], benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], best_performances, benchmarklib="yahpogym")
    return baseline_config_df, prior_config_df, prior_priors_df


def load_performance_data_yahpo_ablation():
    """
    Load the performance data, saved in the filesystem. Do some data_cleaning for lcbench and add regret.
    """

    def _clean_lcbench_performance(df: pd.DataFrame):
        mask = df["scenario"] == "lcbench"
        df.loc[mask, "performance"] = df.loc[mask, "performance"] / 100
        df.loc[mask, "final_performance"] = df.loc[mask, "final_performance"] / 100
        return df

    baseline_table = pd.read_csv(YAHPO_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(YAHPO_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(YAHPO_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(YAHPO_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(YAHPO_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    prior_config_df = prior_config_df[prior_config_df["dynabo"] == False]
    prior_priors_df = prior_priors_df[prior_priors_df["dynabo"] == False]

    prior_ablation_table = pd.read_csv(YAHPO_ABLATION_TABLE_PATH)
    prior_ablation_configs = pd.read_csv(YAHPO_ABLATION_INCUMBENT_PATH)
    prior_ablation_priors = pd.read_csv(YAHPO_ABLATION_PRIOR_PATH)
    prior_ablation_config_df, prior_ablation_priors_df = merge_df(prior_ablation_table, prior_ablation_configs, prior_ablation_priors)
    prior_config_df = pd.concat([prior_config_df, prior_ablation_config_df])
    prior_priors_df = pd.concat([prior_priors_df, prior_ablation_priors_df])

    # For scenario lcbench divide by final_performance and perforamnce by 100
    baseline_config_df = _clean_lcbench_performance(baseline_config_df)
    prior_config_df = _clean_lcbench_performance(prior_config_df)
    prior_priors_df = _clean_lcbench_performance(prior_priors_df)

    best_performances = get_best_performances([baseline_config_df, prior_config_df], benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], best_performances, benchmarklib="yahpogym")
    return baseline_config_df, prior_config_df, prior_priors_df


def plot_yahpo_ablation():
    baseline_config_df, prior_config_df, prior_prior_df = load_performance_data_yahpo_ablation()

    thresholds = [-2, -0.5, -1, 0, 0.5, 1, 2]
    config_dict = {
        "Vanilla BO": baseline_config_df,
    }
    prior_dict = {}

    for threshold in thresholds:
        dynabo_incumbent_df, dynabo_prior_df = filter_prior_approach(
            incumbent_df=prior_config_df,
            prior_df=prior_prior_df,
            select_dynabo=True,
            select_pibo=False,
            with_validating=True,
            prior_validation_method="difference",
            prior_validation_manwhitney_p=None,
            prior_validation_difference_threshold=threshold,
        )
        config_dict[rf"$\tau$={threshold}"] = dynabo_incumbent_df
        prior_dict[rf"$\tau$={threshold}"] = dynabo_prior_df

    dynabo_incumbent_df_without_validation, dynabo_prior_df_without_validation = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict[r"$\tau$=-$\infty$"] = dynabo_incumbent_df_without_validation

    # create_dataset_plots(
    #    config_dict,
    #    prior_dict,
    #    error_bar_type="se",
    #    scenarios=baseline_config_df["scenario"].unique(),
    #    benchmarklib="yahpogym",
    #    base_path="plots/yahpo_ablation/dataset_plots",
    # )

    style_dict = {
        "Vanilla BO": {"color": "black", "marker": "o"},
        r"$\tau$=-$\infty$": {"color": "m", "marker": "s"},
        r"$\tau$=-2": {"color": "tab:orange", "marker": "D"},
        r"$\tau$=-1": {"color": "tab:green", "marker": "v"},
        r"$\tau$=-0.5": {"color": "tab:purple", "marker": "^"},
        r"$\tau$=0": {"color": "tab:brown", "marker": "p"},
        r"$\tau$=0.5": {"color": "tab:pink", "marker": "*"},
        r"$\tau$=2": {"color": "tab:olive", "marker": "d"},
        r"$\tau$=1": {"color": "tab:gray", "marker": "X"},
    }

    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="yahpogym", base_path="plots/yahpo_ablation", ncol=len(style_dict))
    create_scenario_plots(
        config_dict, prior_dict, style_dict, error_bar_type="se", scenarios=baseline_config_df["scenario"].unique(), benchmarklib="yahpogym", base_path="plots/yahpo_ablation", ncol=len(style_dict)
    )


def plot_final_results_yahpo():
    baseline_config_df, prior_config_df, prior_prior_df = load_performance_data_yahpo_ablation()
    dynabo_incumbent_df_with_validation_difference_1, dynabo_prior_df_with_validation_difference_1 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    dynabo_incumbent_df_without_validation, dynabo_prior_df_without_validation = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )
    pibo_incumbent_df_without_validation, pibo_prior_df_without_validation = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict = {
        "DynaBO, accept all priors": dynabo_incumbent_df_without_validation,
        r"DynaBO, accept helpful priors ($\tau$ = -1)": dynabo_incumbent_df_with_validation_difference_1,
        r"$\pi$BO": pibo_incumbent_df_without_validation,
        "Vanilla BO": baseline_config_df,
    }

    prior_dict = {
        "DynaBO, accept all priors": dynabo_prior_df_without_validation,
        "DynaBO, accept helpful priors ($\tau$ = -1)": dynabo_prior_df_with_validation_difference_1,
        r"$\pi$BO": pibo_prior_df_without_validation,
    }

    style_dict = {
        "DynaBO, accept all priors": {"color": "m", "marker": "s"},
        r"DynaBO, accept helpful priors ($\tau$ = -1)": {"color": "tab:green", "marker": "v"},
        r"$\pi$BO": {"color": "tab:cyan", "marker": "d"},
        "Vanilla BO": {"color": "black", "marker": "o"},
    }

    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="yahpogym", base_path="plots/scenario_plots", ncol=len(style_dict))
    create_scenario_plots(
        config_dict, prior_dict, style_dict, error_bar_type="se", scenarios=baseline_config_df["scenario"].unique(), benchmarklib="yahpogym", base_path="plots/scenario_plots", ncol=len(style_dict)
    )


if __name__ == "__main__":
    # plot_yahpo_ablation()
    plot_final_results_yahpo()
