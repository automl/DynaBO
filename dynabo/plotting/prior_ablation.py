import os
import random
from copy import deepcopy
from typing import Dict, List

import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.data_processing.download_all_files import (
    PD1_ABLATION_INCUMBENT_PATH,
    PD1_ABLATION_PRIOR_PATH,
    PD1_ABLATION_TABLE_PATH,
    PD1_BASELINE_INCUMBENT_PATH,
    PD1_BASELINE_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, get_best_performances, merge_df, preprocess_configs, save_fig, set_ax_style


def sample_colors(n):
    """
    Return n unique random Matplotlib color names from the full set.

    Parameters:
        n (int): number of colors to sample (must be <= len(color_names))

    Returns:
        List[str]: randomly sampled color names
    """
    all_named_colors = mcolors.get_named_colors_mapping()

    # Extract just the color names
    color_names = list(all_named_colors.keys())

    if not 0 < n <= len(color_names):
        raise ValueError(f"n must be between 1 and {len(color_names)}")
    return random.sample(color_names, n)


def load_performance_data_mfpbench_ablate_priors(prior_std_denominator: int):
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

    prior_table = pd.read_csv(PD1_ABLATION_TABLE_PATH)
    prior_table = prior_table[prior_table["prior_std_denominator"] == prior_std_denominator]
    prior_configs = pd.read_csv(PD1_ABLATION_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PD1_ABLATION_PRIOR_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    baseline_config_df = invert_performance(baseline_config_df)
    prior_config_df = invert_performance(prior_config_df)
    prior_priors_df = invert_performance(prior_priors_df)

    best_performances = get_best_performances([baseline_config_df, prior_config_df], benchmarklib="mfpbench")
    baseline_config_df, prior_config_df, prior_priors_df = add_regret([baseline_config_df, prior_config_df, prior_priors_df], best_performances, benchmarklib="mfpbench")

    return baseline_config_df, prior_config_df, prior_priors_df


def filter_prior_ablation(
    incumbent_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    prior_number: int,
):
    incumbent_df = incumbent_df[incumbent_df["prior_number"] == prior_number]
    prior_df = prior_df[prior_df["prior_number"] == prior_number]

    return incumbent_df, prior_df


def plot_final_run_prior_ablation(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    style_dict: Dict[str, Dict[str, str]],
    scenario: str,
    dataset: str,
    prior_kind: str,
    ax: plt.Axes,
    benchmarklib: str,
    min_ntrials=1,
    max_ntrials=200,
    error_bar_type: str = "se",
):
    config_dict = deepcopy(config_dict)
    prior_dict = deepcopy(prior_dict)
    # Select relevant data
    config_dict, prior_dict = preprocess_configs(
        config_dict,
        prior_dict,
        scenario,
        dataset,
        prior_kind,
        benchmarklib,
        min_ntrials,
        max_ntrials,
    )

    prior_dict = extract_priors_prior_ablation(config_dict, prior_dict)

    for key, df in config_dict.items():
        color = style_dict[key]["color"]
        sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df, label=key, ax=ax, errorbar=error_bar_type, color=color)

    for key, df in prior_dict.items():
        color = style_dict[key]["color"]
        sns.scatterplot(x="after_n_evaluations", y="regret", data=df, label=key, ax=ax, color=color, marker=style_dict[key]["marker"], s=50)

    return ax


def extract_priors_prior_ablation(config_dict: Dict[str, pd.DataFrame], prior_dict: Dict[str, pd.DataFrame]):
    new_config_dict = deepcopy(config_dict)
    del new_config_dict["Vanilla BO"]

    prior_dict_new = {}
    for key, df in prior_dict.items():
        prior_df = deepcopy(df.iloc[[0]])
        local_config_dict = deepcopy(new_config_dict[key])
        local_config_dict = local_config_dict[["after_n_evaluations", "regret"]].groupby(["after_n_evaluations"]).mean().reset_index()
        first_beating_prior = local_config_dict[local_config_dict["regret"] < prior_df["regret"].iloc[0]]
        if len(first_beating_prior) > 0:
            prior_dict_new[key] = deepcopy(first_beating_prior.iloc[[0]])
        else:
            prior_df["after_n_evaluations"] = 50
            prior_dict_new[key] = prior_df[["after_n_evaluations", "regret"]]

    return prior_dict_new


def create_scenario_plots_prior_ablation(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    style_dict: Dict[str, str],
    error_bar_type: str,
    scenarios: List[str],
    benchmarklib: str,
    base_path: str,
    ncol: int,
    prior_std_denominator: int,
):
    if benchmarklib == "yahpogym":
        min_ntrials = 1
        max_n_trials = 200
    elif benchmarklib == "mfpbench":
        min_ntrials = 1
        max_n_trials = 50
    for scenario in scenarios:
        os.makedirs(f"plots/scenario_plots/{benchmarklib}//regret", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(9, 16), dpi=300)  # Wider and higher resolution
        ax = plot_final_run_prior_ablation(
            config_dict,
            prior_dict,
            style_dict,
            scenario,
            None,
            "dummy_value",
            ax=ax,
            benchmarklib=benchmarklib,
            min_ntrials=min_ntrials,
            max_ntrials=max_n_trials,
            error_bar_type=None,
        )
        set_ax_style(ax, prior_kind="dummy_value", x_label="Number of Evaluations", y_label="Regret")
        # Ax set log scale
        ax.set_yscale("log")
        save_fig(f"{base_path}/{benchmarklib}/regret/{scenario}_std_{prior_std_denominator}.pdf")
        print(f"Saved {scenario}")


def plot_prior_ablation(prior_std_denominator: int):
    baseline_config_df, prior_config_df, prior_prior_df = load_performance_data_mfpbench_ablate_priors(prior_std_denominator)

    for scneario in baseline_config_df["scenario"].unique():
        scenario_config_dict = {}
        scenario_prior_dict = {}

        scenario_config_dict["Vanilla BO"] = baseline_config_df[baseline_config_df["scenario"] == scneario]
        for prior_number in range(55):
            incumbent_df = deepcopy(prior_config_df[(prior_config_df["scenario"] == scneario) & (prior_config_df["prior_number"] == prior_number)])
            prior_df = deepcopy(prior_prior_df[(prior_prior_df["scenario"] == scneario) & (prior_prior_df["prior_number"] == prior_number)])
            if len(prior_df) == 0:
                continue

            scenario_config_dict[f"prior_{prior_number}"] = incumbent_df
            scenario_prior_dict[f"prior_{prior_number}"] = prior_df

        colors = sample_colors(len(scenario_config_dict))

        style_dict = {
            "Vanilla BO": {"color": colors[0], "marker": "o"},
        }

        for prior_number in range(len(scenario_config_dict) - 1):
            style_dict[f"prior_{prior_number}"] = {"color": colors[prior_number + 1], "marker": "o"}

        create_scenario_plots_prior_ablation(
            scenario_config_dict,
            scenario_prior_dict,
            style_dict,
            error_bar_type="se",
            scenarios=[scneario],
            benchmarklib="mfpbench",
            base_path="plots/all_priors_pibo",
            ncol=len(style_dict),
            prior_std_denominator=prior_std_denominator,
        )


if __name__ == "__main__":
    plot_prior_ablation(5)
    plot_prior_ablation(15)
    plot_prior_ablation(100)
    plot_prior_ablation(1000)
    plot_prior_ablation(5000)
