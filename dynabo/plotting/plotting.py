# %%
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.data_processing.download_all_files import BASELINE_INCUMBENT_PATH, BASELINE_TABLE_PATH, PRIOR_INCUMBENT_PATH, PRIOR_PRIORS_PATH, PRIOR_TABLE_PATH


def load_datageneration_data():
    main_table = pd.read_csv("plotting_data/datageneration_medium_hard.csv")
    configs = pd.read_csv("plotting_data/datageneration_incumbent_medium_hard.csv")

    main_table, _ = merge_df(main_table, configs, None)

    max_performances = get_max_performance(baseline_config_df=main_table, prior_config_df=None)
    main_table = add_regret([main_table], max_performances)[0]
    return main_table


def load_final_data():
    baseline_table = pd.read_csv(BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table = pd.read_csv(PRIOR_TABLE_PATH)
    prior_configs = pd.read_csv(PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PRIOR_PRIORS_PATH)
    prior_config_df, prior_prior_df = merge_df(prior_table, prior_configs, prior_priors)

    # For scenario lcbench divide by final_performance and perforamnce by 100
    baseline_config_df["performance"] = baseline_config_df.apply(lambda x: x["performance"] / 100 if x["scenario"] == "lcbench" else x["performance"], axis=1)
    prior_config_df["performance"] = prior_config_df.apply(lambda x: x["performance"] / 100 if x["scenario"] == "lcbench" else x["performance"], axis=1)
    prior_prior_df["performance"] = prior_prior_df.apply(lambda x: x["performance"] / 100 if x["scenario"] == "lcbench" else x["performance"], axis=1)
    baseline_config_df["final_performance"] = baseline_config_df.apply(lambda x: x["final_performance"] / 100 if x["scenario"] == "lcbench" else x["final_performance"], axis=1)
    prior_config_df["final_performance"] = prior_config_df.apply(lambda x: x["final_performance"] / 100 if x["scenario"] == "lcbench" else x["final_performance"], axis=1)
    prior_prior_df["final_performance"] = prior_prior_df.apply(lambda x: x["final_performance"] / 100 if x["scenario"] == "lcbench" else x["final_performance"], axis=1)

    max_performances = get_max_performance(baseline_config_df=baseline_config_df, prior_config_df=prior_config_df)
    baseline_config_df, prior_config_df, prior_prior_df = add_regret([baseline_config_df, prior_config_df, prior_prior_df], max_performances)
    return baseline_config_df, prior_config_df, prior_prior_df


def merge_df(df: pd.DataFrame, incumbents: pd.DataFrame, priors: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    incumbents = incumbents.drop(columns=["ID"])
    incumbent_df = df.merge(incumbents, left_on="ID", right_on="experiment_id")
    if priors is not None:
        priors = priors.drop(columns=["ID"])
        prior_df = df.merge(priors, left_on="ID", right_on="experiment_id")
    else:
        prior_df = None

    return incumbent_df, prior_df


def get_max_performance(baseline_config_df: pd.DataFrame, prior_config_df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """
    Compute the maximum performance, later needed for regret computation for all experiments. If ignore_data_generation is True, the data generation experiments are ignored.

    """
    all_dfs = [baseline_config_df, prior_config_df]
    concat_df = pd.concat(all_dfs)
    max_performances = concat_df.groupby(["scenario", "dataset"])["performance"].max()
    return max_performances.to_dict()


def add_regret(dfs: List[pd.DataFrame], max_performances: Dict[Tuple[str, int], float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the regret to the dataframes
    """
    for df in dfs:
        df["regret"] = df.apply(lambda x: max_performances[(x["scenario"], x["dataset"])] - x["performance"], axis=1)
        df["final_regret"] = df.apply(lambda x: max_performances[(x["scenario"], x["dataset"])] - x["final_performance"], axis=1)
    return dfs


def split_df(prior_config_df: pd.DataFrame, prior_prior_df: pd.DataFrame):
    dynabo_incumbent_df = prior_config_df[prior_config_df["dynabo"] == True]
    dynabo_prior_df = prior_prior_df[prior_prior_df["dynabo"] == True]
    pibo_incumbent_df = prior_config_df[prior_config_df["pibo"] == True]
    pibo_prior_df = prior_prior_df[prior_prior_df["pibo"] == True]

    return dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df


def quantile_ci(data):
    return np.percentile(data, [5, 95.5])


def plot_datageneration_run(
    data: pd.DataFrame,
    scenario: str,
    ax: plt.Axes,
    min_ntrials=1,
    max_ntrials=5000,
    error_bar_type: str = "se",
):
    random_incumbent_data = data[(data["scenario"] == scenario) & (data["random"] == True)]
    smac_incumbent_data = data[(data["scenario"] == scenario) & (data["random"] == False)]

    df_dict = {
        "random": random_incumbent_data,
        "smac": smac_incumbent_data,
    }

    df_dict = extract_incumbent_steps(df_dict=df_dict, min_ntrials=min_ntrials, max_ntrials=max_ntrials)

    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["random"], label="random", ax=ax, errorbar=error_bar_type)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["smac"], label="smac", ax=ax, errorbar=error_bar_type)
    ax.set_ylabel("Regret")

    # Check highest performacne after 10 trials
    return ax


def plot_final_run(
    baseline_data: pd.DataFrame,
    dynabo_incumbent_data: pd.DataFrame,
    dynabo_prior_data: pd.DataFrame,
    pibo_incumbent_data: pd.DataFrame,
    pibo_prior_data: pd.DataFrame,
    scenario: str,
    dataset: str,
    prior_kind: str,
    use_rejection: bool,
    ax: plt.Axes,
    min_ntrials=1,
    max_ntrials=200,
    error_bar_type: str = "se",
):
    # Select relevant data
    df_dict = preprocess(baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, prior_kind, use_rejection, ax, min_ntrials, max_ntrials)

    # sns.scatterplot(x="after_n_evaluations", y="regret", ax=ax, data=relevant_dynabo_priors)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["baseline"], label="baseline", ax=ax, errorbar=error_bar_type)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["dynabo_incumbents"], label="dynabo", ax=ax, errorbar=error_bar_type)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["pibo_incumbents"], label="pibo", ax=ax, errorbar=error_bar_type)
    ax.set_ylabel("Regret")

    # Check highest performacne after 10 trials
    highest_regret = max([df_dict[i][df_dict["baseline"]["after_n_evaluations"] == 10]["regret"].mean() for i in df_dict.keys()])
    ax.set_ylim(0, highest_regret * 1.1)

    return ax


def plot_cdf(
    baseline_data: pd.DataFrame,
    dynabo_incumbent_data: pd.DataFrame,
    dynabo_prior_data: pd.DataFrame,
    pibo_incumbent_data: pd.DataFrame,
    pibo_prior_data: pd.DataFrame,
    scenario: str,
    dataset: str,
    use_rejection: bool,
    ax: plt.Axes,
):
    relevant_baseline, relevant_dynabo_incumbents_good, _, relevant_pibo_incumbents_good, _ = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, "good", use_rejection
    )
    _, relevant_dynabo_incumbents_medium, _, relevant_pibo_incumbents_medium, _ = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, "medium", use_rejection
    )
    _, relevant_dynabo_incumbents_misleading, _, relevant_pibo_incumbents_misleading, _ = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, "misleading", use_rejection
    )

    sns.ecdfplot(x="final_regret", data=relevant_baseline, ax=ax, label="baseline")
    sns.ecdfplot(x="final_regret", data=relevant_dynabo_incumbents_good, ax=ax, label="dynabo_good")
    sns.ecdfplot(x="final_regret", data=relevant_dynabo_incumbents_medium, ax=ax, label="dynabo_medium")
    sns.ecdfplot(x="final_regret", data=relevant_dynabo_incumbents_misleading, ax=ax, label="dynabo_misleading")
    ax.set_ylabel("CDF")
    ax.set_xlabel("Regret")
    return ax


def preprocess(
    baseline_data: pd.DataFrame,
    dynabo_incumbent_data: pd.DataFrame,
    dynabo_prior_data: pd.DataFrame,
    pibo_incumbent_data: pd.DataFrame,
    pibo_prior_data: pd.DataFrame,
    scenario: str,
    dataset: str,
    prior_kind: str,
    use_rejection: bool,
    ax: plt.Axes,
    min_ntrials=1,
    max_ntrials=200,
):
    relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_priors, relevant_pibo_incumbents, relevant_pibo_priors = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, prior_kind, use_rejection
    )

    df_dict = {
        "baseline": relevant_baseline,
        "dynabo_incumbents": relevant_dynabo_incumbents,
        "pibo_incumbents": relevant_pibo_incumbents,
    }

    df_dict = extract_incumbent_steps(df_dict=df_dict, min_ntrials=min_ntrials, max_ntrials=max_ntrials)

    return df_dict


def select_relevant_data(
    baseline_data: pd.DataFrame,
    dynabo_incumbent_data: pd.DataFrame,
    dynabo_prior_data,
    pibo_incumbent_data: pd.DataFrame,
    pibo_prior_data: pd.DataFrame,
    scenario: str,
    dataset: str,
    prior_kind: str,
    use_rejection: bool,
):
    if scenario is None:
        relevant_baseline = baseline_data[baseline_data["incumbent"] == 1]
        relevant_dynabo_incumbents = dynabo_incumbent_data[
            (dynabo_incumbent_data["prior_kind"] == prior_kind) & (dynabo_incumbent_data["validate_prior"] == use_rejection) & (dynabo_incumbent_data["incumbent"] == 1)
        ]
        relevant_dynabo_prior = dynabo_prior_data[(dynabo_prior_data["prior_kind"] == prior_kind) & (dynabo_prior_data["validate_prior"] == use_rejection)]
        relevant_pibo_incumbents = pibo_incumbent_data[(pibo_incumbent_data["prior_kind"] == prior_kind) & (pibo_incumbent_data["incumbent"] == 1)]
        relevant_pibo_prior = pibo_prior_data[(pibo_prior_data["prior_kind"] == prior_kind) & (pibo_prior_data["validate_prior"] == use_rejection)]

    elif dataset is not None:
        relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario) & (baseline_data["dataset"] == dataset) & baseline_data["incumbent"] == 1]
        relevant_dynabo_incumbents = dynabo_incumbent_data[
            (dynabo_incumbent_data["scenario"] == scenario)
            & (dynabo_incumbent_data["dataset"] == dataset)
            & (dynabo_incumbent_data["prior_kind"] == prior_kind)
            & (dynabo_incumbent_data["validate_prior"] == use_rejection)
            & (dynabo_incumbent_data["incumbent"] == 1)
        ]
        relevant_dynabo_prior = dynabo_prior_data[
            (dynabo_prior_data["scenario"] == scenario)
            & (dynabo_prior_data["dataset"] == dataset)
            & (dynabo_prior_data["prior_kind"] == prior_kind)
            & (dynabo_prior_data["validate_prior"] == use_rejection)
        ]
        relevant_pibo_incumbents = pibo_incumbent_data[
            (pibo_incumbent_data["scenario"] == scenario) & (pibo_incumbent_data["dataset"] == dataset) & (pibo_incumbent_data["prior_kind"] == prior_kind) & (pibo_incumbent_data["incumbent"] == 1)
        ]
        relevant_pibo_prior = pibo_prior_data[
            (pibo_prior_data["scenario"] == scenario) & (pibo_prior_data["dataset"] == dataset) & (pibo_prior_data["prior_kind"] == prior_kind) & (pibo_prior_data["validate_prior"] == use_rejection)
        ]
    else:
        relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario) & baseline_data["incumbent"] == 1]
        relevant_dynabo_incumbents = dynabo_incumbent_data[
            (dynabo_incumbent_data["scenario"] == scenario)
            & (dynabo_incumbent_data["prior_kind"] == prior_kind)
            & (dynabo_incumbent_data["validate_prior"] == use_rejection)
            & (dynabo_incumbent_data["incumbent"] == 1)
        ]
        relevant_dynabo_prior = dynabo_prior_data[
            (dynabo_prior_data["scenario"] == scenario) & (dynabo_prior_data["prior_kind"] == prior_kind) & (dynabo_prior_data["validate_prior"] == use_rejection)
        ]
        relevant_pibo_incumbents = pibo_incumbent_data[(pibo_incumbent_data["scenario"] == scenario) & (pibo_incumbent_data["prior_kind"] == prior_kind) & (pibo_incumbent_data["incumbent"] == 1)]
        relevant_pibo_prior = pibo_prior_data[(pibo_prior_data["scenario"] == scenario) & (pibo_prior_data["prior_kind"] == prior_kind) & (pibo_prior_data["validate_prior"] == use_rejection)]
    return relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_prior, relevant_pibo_incumbents, relevant_pibo_prior


def extract_incumbent_steps(df_dict: Dict[str, pd.DataFrame], min_ntrials: int, max_ntrials: int):
    full_range = pd.DataFrame({"after_n_evaluations": range(min_ntrials, max_ntrials + 1)})

    # Step 1: Iterate over all DataFrames
    for key in df_dict.keys():
        df = df_dict[key]
        local = list()
        # Ensure sorting
        df.sort_values(["scenario", "dataset", "seed", "after_n_evaluations"], inplace=True)
        # Step 2: Iterate over each scenario
        for scenario in df["scenario"].unique():
            scenario_mask = df["scenario"] == scenario
            scenario_df = df[scenario_mask]

            # Step 3: Iterate over each dataset in the current scenario
            for dataset in scenario_df["dataset"].unique():
                dataset_mask = scenario_mask & (df["dataset"] == dataset)
                dataset_df = df[dataset_mask]

                # Step 4: Iterate over each seed in the current dataset
                for seed in dataset_df["seed"].unique():
                    seed_mask = dataset_mask & (df["seed"] == seed)

                    # Merge the full range with the current group to ensure all `after_n_evaluations` are included
                    merged_df = full_range.merge(df.loc[seed_mask], on="after_n_evaluations", how="left")

                    merged_df["regret"] = merged_df["regret"].ffill()
                    merged_df_final = pd.DataFrame(columns=["scenario", "dataset", "seed", "after_n_evaluations", "regret"])
                    merged_df_final["after_n_evaluations"] = merged_df["after_n_evaluations"]
                    merged_df_final["regret"] = merged_df["regret"]
                    merged_df_final["scenario"] = scenario
                    merged_df_final["dataset"] = dataset
                    merged_df_final["seed"] = seed

                    local.append(merged_df_final)

        # Concatenate the local list to a DataFrame
        df_dict[key] = pd.concat(local)
    return df_dict


def create_dataset_plots(
    baseline_config_df: pd.DataFrame, dynabo_incumbent_df: pd.DataFrame, dynabo_prior_df: pd.DataFrame, pibo_incumbent_df: pd.DataFrame, pibo_prior_df: pd.DataFrame, error_bar_type: str
):
    for scenario in baseline_config_df["scenario"].unique():
        scenario_df = baseline_config_df[baseline_config_df["scenario"] == scenario]
        for dataset in scenario_df["dataset"].unique():
            os.makedirs(f"plots/dataset_plots/regret/{scenario}", exist_ok=True)
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)  # Wider and higher resolution
            axs = axs.flatten()
            plot_number = 0
            for prior_kind in ["good", "medium", "misleading"]:
                ax = axs[plot_number]
                ax = plot_final_run(
                    baseline_config_df,
                    dynabo_incumbent_df,
                    dynabo_prior_df,
                    pibo_incumbent_df,
                    pibo_prior_df,
                    scenario,
                    dataset,
                    prior_kind,
                    ax=ax,
                    use_rejection=False,
                    min_ntrials=1,
                    max_ntrials=200,
                    error_bar_type=error_bar_type,
                )
                plot_number += 1
                set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Trials", y_label="Regret")
            set_fig_style(fig, axs, f"Regret on {scenario} {dataset}")
            save_fig(f"plots/dataset_plots/regret/{scenario}/{dataset}.pdf")

            os.makedirs(f"plots/dataset_plots/cdf/{scenario}", exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(18, 6), dpi=300)  # Wider and higher resolution
            # ax = axs.flatten()
            ax = plot_cdf(
                baseline_config_df,
                dynabo_incumbent_df,
                dynabo_prior_df,
                pibo_incumbent_df,
                pibo_prior_df,
                scenario,
                dataset,
                use_rejection=False,
                ax=ax,
            )
            set_ax_style(ax, prior_kind=prior_kind, x_label="Regret", y_label="CDF")
            set_fig_style(fig, f"CDF of Regret on {scenario} {dataset}")
            save_fig(f"plots/dataset_plots/cdf/{scenario}/{dataset}.pdf")


def create_scenario_plots(
    baseline_config_df: pd.DataFrame, dynabo_incumbent_df: pd.DataFrame, dynabo_prior_df: pd.DataFrame, pibo_incumbent_df: pd.DataFrame, pibo_prior_df: pd.DataFrame, error_bar_type: str
):
    for scenario in baseline_config_df["scenario"].unique():
        os.makedirs("plots/scenario_plots/regret", exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)  # Wider and higher resolution
        axs = axs.flatten()
        plot_number = 0
        for prior_kind in ["good", "medium", "misleading"]:
            ax = axs[plot_number]
            ax = plot_final_run(
                baseline_config_df,
                dynabo_incumbent_df,
                dynabo_prior_df,
                pibo_incumbent_df,
                pibo_prior_df,
                scenario,
                None,
                prior_kind,
                ax=ax,
                use_rejection=False,
                min_ntrials=1,
                max_ntrials=200,
                error_bar_type=error_bar_type,
            )
            plot_number += 1
            set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")
        set_fig_style(fig, axs, f"Average regret on {scenario}")
        save_fig(f"plots/scenario_plots/regret/{scenario}.pdf")
        print(f"Saved {scenario}.png")

        os.makedirs("plots/scenario_plots/cdf", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(18, 6), dpi=300)  # Wider and higher resolution
        # ax = axs.flatten()
        ax = plot_cdf(
            baseline_config_df,
            dynabo_incumbent_df,
            dynabo_prior_df,
            pibo_incumbent_df,
            pibo_prior_df,
            scenario,
            None,
            use_rejection=False,
            ax=ax,
        )
        set_ax_style(ax, prior_kind=prior_kind, x_label="Regret", y_label="CDF")
        set_fig_style(fig, axs, f"CDF of Regret on {scenario}")
        save_fig(f"plots/scenario_plots/cdf/{scenario}.pdf")


def create_overall_plot(
    baseline_config_df: pd.DataFrame, dynabo_incumbent_df: pd.DataFrame, dynabo_prior_df: pd.DataFrame, pibo_incumbent_df: pd.DataFrame, pibo_prior_df: pd.DataFrame, error_bar_type: str
):
    os.makedirs("plots/scenario_plots/regret", exist_ok=True)

    # Improved figure layout
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)  # Wider and higher resolution
    axs = axs.flatten()

    # Define colors for better differentiation
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    plot_number = 0
    for prior_kind, color in zip(["good", "medium", "misleading"], colors):
        ax = axs[plot_number]

        # Call the plotting function
        ax = plot_final_run(
            baseline_config_df,
            dynabo_incumbent_df,
            dynabo_prior_df,
            pibo_incumbent_df,
            pibo_prior_df,
            None,
            None,
            prior_kind,
            ax=ax,
            use_rejection=False,
            min_ntrials=1,
            max_ntrials=200,
            error_bar_type=error_bar_type,
        )

        set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")

        plot_number += 1

    set_fig_style(fig, axs, "Overall Regret Across Different Priors")

    save_fig("plots/scenario_plots/regret/overall.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(18, 6), dpi=300)  # Wider and higher resolution
    # ax = axs.flatten()
    ax = plot_cdf(
        baseline_config_df,
        dynabo_incumbent_df,
        dynabo_prior_df,
        pibo_incumbent_df,
        pibo_prior_df,
        None,
        None,
        use_rejection=False,
        ax=ax,
    )
    set_ax_style(ax, prior_kind=prior_kind, x_label="Regret", y_label="CDF")
    save_fig("plots/scenario_plots/cdf/overall.pdf")


def set_ax_style(ax, prior_kind: str, x_label, y_label):
    # Remove ax legend
    ax.legend().remove()

    # Improve title aesthetics
    ax.set_title(
        f"Prior: {prior_kind.capitalize()}",
        fontsize=30,
        fontweight="bold",
    )

    # Enhance axes labels and grid
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlabel(x_label, fontsize=25, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=25, fontweight="bold")


def set_fig_style(fig, axs, title: str):
    fig.suptitle(title, fontsize=25, fontweight="bold", y=1)

    # Extract all plotted lines from axs and only keep unique lines
    label_line_dict = {line.get_label(): line for ax in axs for line in ax.get_lines()}
    labels, lines = zip(*label_line_dict.items())

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=20,
    )

    # Adjust layout for better spacing
    fig.tight_layout()


def save_fig(path: str):
    # Save the figure with high quality
    plt.savefig(path, bbox_inches="tight", dpi=300, transparent=True)

    plt.close()


def remove_weird_datasets(
    baseline_config_df: pd.DataFrame, dynabo_incumbent_df: pd.DataFrame, dynabo_prior_df: pd.DataFrame, pibo_incumbent_df: pd.DataFrame, pibo_prior_df: pd.DataFrame
) -> List[Tuple[str, int]]:
    # remove datasets that do not have 30 priros
    original_dataset_counts = dynabo_prior_df.groupby(["scenario", "dataset"]).size().reset_index(name="count")
    dataset_counts = dynabo_prior_df[dynabo_prior_df["no_superior_configuration"] == 0].groupby(["scenario", "dataset"]).size().reset_index(name="count")
    datasets = pd.merge(original_dataset_counts, dataset_counts, how="inner", on=["scenario", "dataset", "count"])[["scenario", "dataset"]]

    baseline_config_df = pd.merge(left=datasets, right=baseline_config_df, on=["scenario", "dataset"])
    dynabo_incumbent_df = pd.merge(left=datasets, right=dynabo_incumbent_df, on=["scenario", "dataset"])
    dynabo_prior_df = pd.merge(left=datasets, right=dynabo_prior_df, on=["scenario", "dataset"])
    pibo_incumbent_df = pd.merge(left=datasets, right=pibo_incumbent_df, on=["scenario", "dataset"])
    pibo_prior_df = pd.merge(left=datasets, right=pibo_prior_df, on=["scenario", "dataset"])
    return baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df


def plot_final_results():
    baseline_config_df, prior_config_df, prior_prior_df = load_final_data()
    dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df = split_df(prior_config_df=prior_config_df, prior_prior_df=prior_prior_df)
    baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df = remove_weird_datasets(
        baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df
    )
    # create_dataset_plots(baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df, error_bar_type="se")
    create_scenario_plots(baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df, error_bar_type="se")
    create_overall_plot(baseline_config_df, dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df, error_bar_type="se")


def plot_datageneration():
    data = load_datageneration_data()
    fig, axs = plt.subplots(1, len(data["scenario"].unique()), figsize=(32, 18), dpi=300)
    axs = axs.flatten()
    for scenario, ax in zip(data["scenario"].unique(), axs):
        plot_datageneration_run(data, scenario, ax)
        ax.set_title(scenario)
    fig.legend()
    save_fig(
        "plots/data_generation/joined.pdf",
    )


if __name__ == "__main__":
    plot_final_results()
    # plot_datageneration()
