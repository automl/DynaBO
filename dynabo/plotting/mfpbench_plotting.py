import pandas as pd

from dynabo.data_processing.download_all_files import (
    PD1_BASELINE_INCUMBENT_PATH,
    PD1_BASELINE_TABLE_PATH,
    PD1_PRIOR_INCUMBENT_PATH,
    PD1_PRIOR_PRIORS_PATH,
    PD1_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, create_overall_plot, create_scenario_plots, filter_prior_approach, get_best_performances, merge_df


def load_performance_data_mfpbench():
    """
    Load the performance data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """

    def invert_performance(df: pd.DataFrame):
        df["performance"] = df["performance"] * -1
        df["final_performance"] = df["final_performance"] * -1
        return df

    baseline_table = pd.read_csv(PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    baseline_config_df = invert_performance(baseline_config_df)
    prior_config_df = invert_performance(prior_config_df)
    prior_priors_df = invert_performance(prior_priors_df)

    best_performances = get_best_performances([baseline_config_df, prior_config_df], benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], best_performances, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_mfpbench():
    baseline_config_df, prior_config_df, prior_prior_df = load_performance_data_mfpbench()

    dynabo_incumbent_df_chance_01_threshold, dynabo_prior_df_chance_01_threshold = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_chance_theta_choices=0.01,
        with_validating=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    dynabo_incumbent_df_chance_015_threshold, dynabo_prior_df_chance_015_threshold = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_chance_theta_choices=0.015,
        with_validating=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    dynabo_incumbent_df_without_validation_chance_01, dynabo_prior_df_without_validation_chance_01 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_chance_theta_choices=0.01,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    dynabo_incumbent_df_without_validation_chance_015, dynabo_prior_df_without_validation_chance_015 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_chance_theta_choices=0.015,
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
        prior_chance_theta_choices=None,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict = {
        "DynaBO, accept all priors  chance 0.1": dynabo_incumbent_df_without_validation_chance_01,
        "DynaBO, accept all priors  chance 0.15": dynabo_incumbent_df_without_validation_chance_015,
        "DynaBO, accept helpful priors chance 0.1": dynabo_incumbent_df_chance_01_threshold,
        "DynaBO, accept helpful priors chance 0.15": dynabo_incumbent_df_chance_015_threshold,
        r"$\pi$BO": pibo_incumbent_df_without_validation,
        "Vanilla BO": baseline_config_df,
    }

    prior_dict = {
        "DynaBO, accept all priors chance 0.1": dynabo_prior_df_without_validation_chance_01,
        "DynaBO, accept all priors chance 0.15": dynabo_prior_df_without_validation_chance_015,
        "DynaBO, accept helpful priors chance 0.1": dynabo_prior_df_chance_01_threshold,
        "DynaBO, accept helpful priors chance 0.15": dynabo_prior_df_chance_015_threshold,
        r"$\pi$BO": pibo_prior_df_without_validation,
    }

    style_dict = {
        "DynaBO, accept all priors  chance 0.1": {"color": "darkviolet", "marker": "s"},
        "DynaBO, accept all priors  chance 0.15": {"color": "magenta", "marker": "s"},
        "DynaBO, accept helpful priors chance 0.1": {"color": "forestgreen", "marker": "v"},
        "DynaBO, accept helpful priors chance 0.15": {"color": "limegreen", "marker": "v"},
        r"$\pi$BO": {"color": "deepskyblue", "marker": "d"},
        "Vanilla BO": {"color": "black", "marker": "o"},
    }

    # create_dataset_plots(
    #    config_dict=config_dict,
    #    prior_dict=prior_dict,
    #    error_bar_type="se",
    #    scenarios=baseline_config_df["scenario"].unique(),
    # )

    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="mfpbench", base_path="plots/scenario_plots", ncol=len(style_dict))
    create_scenario_plots(
        config_dict, prior_dict, style_dict, error_bar_type="se", scenarios=baseline_config_df["scenario"].unique(), benchmarklib="mfpbench", base_path="plots/scenario_plots", ncol=len(style_dict)
    )


if __name__ == "__main__":
    plot_final_results_mfpbench()
