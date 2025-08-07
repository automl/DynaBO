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
    # TODO extract the stuff that is scurrently in the table and plot the different appraochse agaisnt one another
    baseline_perfect_incumbent_df_decay_25, baseline_perfect_prior_df_decay_25 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=25,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        prior_validation_method="baseline_perfect",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    baseline_perfect_incumbent_df_decay_50, baseline_perfect_prior_df_decay_50 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=50,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        prior_validation_method="baseline_perfect",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    threshold_incumbent_df_decay_25, threshold_prior_df_decay_25 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=25,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    threshold_incumbent_df_decay_50, threshold_prior_df_decay_50 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=50,
        prior_static_position=True,
        prior_every_n_trials=10,
        validate_prior=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    pibo_incumbent_df, pibo_prior_df = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        prior_decay_enumerator=50,
        prior_static_position=None,
        prior_every_n_trials=None,
        validate_prior=None,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        r"$\pi$BO": pibo_incumbent_df,
        "DynaBO, perfect validation, decay 25": baseline_perfect_incumbent_df_decay_25,
        "DynaBO, perfect validation, decay 50": baseline_perfect_incumbent_df_decay_50,
        "DynaBO, threshold validation, decay 25": threshold_incumbent_df_decay_25,
        "DynaBO, threshold validation, decay 50": threshold_incumbent_df_decay_50,
    }
    prior_dict = {
        "Vanilla BO": prior_config_df,
        r"$\pi$BO": pibo_prior_df,
        "DynaBO, perfect validation, decay 25": baseline_perfect_prior_df_decay_25,
        "DynaBO, perfect validation, decay 50": baseline_perfect_prior_df_decay_50,
        "DynaBO, threshold validation, decay 25": threshold_prior_df_decay_25,
        "DynaBO, threshold validation, decay 50": threshold_prior_df_decay_50,
    }

    style_dict = {
        "Vanilla BO": {"color": "black", "marker": "o"},
        r"$\pi$BO": {"color": "deepskyblue", "marker": "d"},
        "DynaBO, perfect validation, decay 25": {"color": "darkviolet", "marker": "s"},
        "DynaBO, perfect validation, decay 50": {"color": "magenta", "marker": "s"},
        "DynaBO, threshold validation, decay 25": {"color": "forestgreen", "marker": "v"},
        "DynaBO, threshold validation, decay 50": {"color": "limegreen", "marker": "v"},
    }

    # style_dict = {
    #    "DynaBO, accept all priors  chance 0.1": {"color": "darkviolet", "marker": "s"},
    #    "DynaBO, accept all priors  chance 0.15": {"color": "magenta", "marker": "s"},
    #    "DynaBO, accept helpful priors chance 0.1": {"color": "forestgreen", "marker": "v"},
    #    "DynaBO, accept helpful priors chance 0.15": {"color": "limegreen", "marker": "v"},
    #    r"$\pi$BO": {"color": "deepskyblue", "marker": "d"},
    #    "Vanilla BO": {"color": "black", "marker": "o"},
    # }

    # create_dataset_plots(
    #    config_dict=config_dict,
    #    prior_dict=prior_dict,
    #    error_bar_type="se",
    #    scenarios=baseline_config_df["scenario"].unique(),
    # )

    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchnmarklib="mfpbench", base_path="plots/prior_rejection_ablation", ncol=len(style_dict))
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
        benchmarklib="mfpbench",
        base_path="plots/prior_rejection_ablation",
        ncol=len(style_dict),
    )


if __name__ == "__main__":
    plot_final_results_mfpbench()
