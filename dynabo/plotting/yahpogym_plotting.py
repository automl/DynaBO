import pandas as pd

from dynabo.data_processing.download_all_files import (
    YAHPO_BASELINE_TABLE_PATH,
    YAHPO_BASELINE_INCUMBENT_PATH,
    YAHPO_PRIOR_TABLE_PATH,
    YAHPO_PRIOR_INCUMBENT_PATH ,
    YAHPO_PRIOR_PRIORS_PATH ,

)
from dynabo.plotting.plotting_utils import add_regret, create_overall_plot, create_scenario_plots, filter_prior_approach, get_min_costs, merge_df
import seaborn as sns


def load_cost_data_yahpogym(surrogate:str):
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
        baseline_config_df.loc[baseline_config_df["scenario"] == "lcbench", "cost"] /= 100
        prior_config_df.loc[prior_config_df["scenario"] == "lcbench", "cost"] /= 100

    else:
        raise ValueError("GP nto used for Yahpo.")

    min_costs = get_min_costs(benchmarklib="yahpogym")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="yahpogym")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_yahpogym(surrogate:str):
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_yahpogym(surrogate = surrogate)
    accept_all_priors_configs, accept_all_priors_priors = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        prior_decay_enumerator=20,
        prior_std_denominator=5,
        prior_static_position=True,
        prior_every_n_trials=40,
        validate_prior=False,
        n_prior_based_samples=0, 
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )
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
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        #"DynaBO, accept all priors": accept_all_priors_configs,
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
        benchmarklib="yahpogym",
        base_path=f"plots/final_result_plots/{surrogate}",
        ncol=4,
    )
    #create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="yahpogym", base_path=f"plots/final_result_plots/{surrogate}", ncol=len(style_dict))


if __name__ == "__main__":
    plot_final_results_yahpogym("rf")