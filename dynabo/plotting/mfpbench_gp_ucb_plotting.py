import pandas as pd

from dynabo.data_processing.download_all_files import (
    GP_UCB_PD1_BASELINE_INCUMBENT_PATH,
    GP_UCB_PD1_BASELINE_TABLE_PATH,
    GP_UCB_PD1_PRIOR_INCUMBENT_PATH,
    GP_UCB_PD1_PRIOR_PRIORS_PATH,
    GP_UCB_PD1_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_regret,
    create_all_scenarios_plot,
    create_overall_plot,
    create_scenario_plots,
    filter_prior_approach,
    get_min_costs,
    merge_df,
)

BASE_PATH = "plots/final_result_plots/gp_ucb"


def load_cost_data_mfpbench_gp_ucb():
    """
    Load the cost data of the pd1 GP + confidence bound experiments, saved in the filesystem, and add regret.
    """
    baseline_table = pd.read_csv(GP_UCB_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(GP_UCB_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(GP_UCB_PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(GP_UCB_PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(GP_UCB_PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def plot_final_results_mfpbench_gp_ucb():
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench_gp_ucb()
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

    # Only plot scenarios for which every approach has runs. `extract_incumbent_steps` cannot handle an
    # approach without data, so a scenario that one approach has not been run on yet is skipped instead.
    all_scenarios = sorted(prior_config_df["scenario"].unique())
    scenarios = [scenario for scenario in all_scenarios if all((df["scenario"] == scenario).any() for df in config_dict.values())]
    for scenario in set(all_scenarios) - set(scenarios):
        missing = [name for name, df in config_dict.items() if not (df["scenario"] == scenario).any()]
        print(f"Skipping scenario {scenario}: no runs for {missing}.")

    # Restrict the aggregated plots to the same scenarios, so that the approaches are averaged over the
    # same set of scenarios. This is a no-op once every approach has been run on all scenarios.
    config_dict = {name: df[df["scenario"].isin(scenarios)] for name, df in config_dict.items()}
    prior_dict = {name: df[df["scenario"].isin(scenarios)] for name, df in prior_dict.items()}

    # Regret plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=scenarios,
        benchmarklib="mfpbench",
        base_path=BASE_PATH,
        ncol=4,
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=BASE_PATH, ncol=len(style_dict))
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=scenarios,
        benchmarklib="mfpbench",
        base_path=BASE_PATH,
        ncol=4,
    )

    # Cost plots
    create_scenario_plots(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=scenarios,
        benchmarklib="mfpbench",
        base_path=BASE_PATH,
        ncol=4,
        y_column="cost",
    )
    create_overall_plot(config_dict, prior_dict, style_dict, error_bar_type="se", benchmarklib="mfpbench", base_path=BASE_PATH, ncol=len(style_dict), y_column="cost")
    create_all_scenarios_plot(
        config_dict,
        prior_dict,
        style_dict,
        error_bar_type="se",
        scenarios=scenarios,
        benchmarklib="mfpbench",
        base_path=BASE_PATH,
        ncol=4,
        y_column="cost",
    )


if __name__ == "__main__":
    plot_final_results_mfpbench_gp_ucb()
