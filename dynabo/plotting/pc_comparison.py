import pandas as pd
import os
from dynabo.data_processing.download_all_files import (
    RF_PD1_BASELINE_INCUMBENT_PATH,
    RF_PD1_BASELINE_TABLE_PATH,
    RF_PD1_PRIOR_INCUMBENT_PATH,
    RF_PD1_PRIOR_PRIORS_PATH,
    RF_PD1_PRIOR_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_regret,
    create_pc_comparison,
    filter_prior_approach,
    get_min_costs,
    merge_df,
)
import numpy as np

def load_cost_data_mfpbench():
    """
    Load the cost data for pd1, saved in the filesystem. Do some data cleaning for lcbench and add regret.
    """
    baseline_table = pd.read_csv(RF_PD1_BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(RF_PD1_BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(RF_PD1_PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(RF_PD1_PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(RF_PD1_PRIOR_PRIORS_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)
    min_costs = get_min_costs(benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def load_cost_data_pc():
    base_path = "plotting_data/pc_data"

    baseline_files = list()
    dist_prior_files = list()
    point_prior_files = list()
    for pc_approach in os.listdir(base_path):
        dir_path = os.path.join(base_path, pc_approach)
        for scneario in os.listdir(dir_path):
            scenario_path = os.path.join(dir_path, scneario)
            if pc_approach == "no_prior":
                for file in os.listdir(scenario_path):
                    file_path = os.path.join(scenario_path, file)
                    df = pd.read_csv(file_path)
                    df ["val_performance"] = - df["val_performance"]
                    # make val performance cumin of val performance
                    df["val_performance"] = df["val_performance"].cummin()
                    df["scenario"] = scneario
                    df = df.rename(columns={"Unnamed: 0": "after_n_evaluations"})
                    baseline_files.append(df)

            else:
                for prior_kind in os.listdir(scenario_path):
                    for file in os.listdir(os.path.join(scenario_path, prior_kind)):
                        file_path = os.path.join(scenario_path, prior_kind, file)
                        df = pd.read_csv(file_path)
                        df ["val_performance"] = - df["val_performance"]
                        # make val performance cumin of val performance
                        df["val_performance"] = df["val_performance"].cummin()
                        df["scenario"] = scneario
                        df["prior_kind"] = prior_kind
                        df = df.rename(columns={"Unnamed: 0": "after_n_evaluations"})
                        if pc_approach == "dist_prior":
                            dist_prior_files.append(df)
                        elif pc_approach == "point_prior":
                            point_prior_files.append(df)
    point_prior_df = pd.concat(point_prior_files, ignore_index=True)
    baseline_df = pd.concat(baseline_files, ignore_index=True)
    dist_prior_df = pd.concat(dist_prior_files, ignore_index=True)
    min_costs = get_min_costs(benchmarklib="mfpbench")

    baseline_df["regret"] = baseline_df.apply(lambda row: row["val_performance"] - min_costs[row["scenario"]], axis=1)
    dist_prior_df["regret"] = dist_prior_df.apply(lambda row: row["val_performance"] - min_costs[row["scenario"]], axis=1)
    point_prior_df["regret"] = point_prior_df.apply(lambda row: row["val_performance"] - min_costs[row["scenario"]], axis=1)

    return baseline_df, dist_prior_df, point_prior_df



def plot_pc_comparison():
    pc_baseline, pc_dist_prior, pc_point_prior = load_cost_data_pc()
    baseline_config_df, prior_config_df, prior_prior_df = load_cost_data_mfpbench()
    
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
        remove_old_priors = False,
        prior_decay="linear",
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        # "DynaBO, accept all priors": accept_all_priors_configs,
        "DynaBO": dynabo_incumbent_df,
        "PC no Prior": pc_baseline,
        "PC Point-Prior": pc_point_prior,
        "PC Dist-Prior": pc_dist_prior,
    }
    prior_dict = {
        # "DynaBO, accept all priors": accept_all_priors_priors,
        "DynaBO": dynabo_prior_df,
    }

    style_dict = {
        "Vanilla BO": {"color": "#000000", "marker": "o", "linestyle": (0, ())},  # Black, solid
        "DynaBO": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},  # Pink, dash-dot dense
        "PC no Prior": {"color": "#0072B2", "marker": "s", "linestyle": (0, (5, 5))},  # Blue, dashed
        "PC Point-Prior": {"color": "#F0E442", "marker": "^", "linestyle": (0, (1, 3))},  # Yellow, dash-dot sparse
        "PC Dist-Prior": {"color": "#009E73", "marker": "D", "linestyle": (0, (3, 5))},  #
    }
    create_pc_comparison(
        config_dict,
        prior_dict,
        baseline_config_df["scenario"].unique(),
        style_dict,
        error_bar_type="se",
        benchmarklib="mfpbench",
        base_path="plots/pc_comparison_new",
        ncol=4,
    )


if __name__ == "__main__":
    plot_pc_comparison()
