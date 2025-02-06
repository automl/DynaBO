# %%
import ast
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.plotting.download_all_files import BASELINE_INCUMBENT_PATH, BASELINE_TABLE_PATH, PRIOR_INCUMBENT_PATH, PRIOR_PRIORS_PATH, PRIOR_TABLE_PATH


def merge_df(df: pd.DataFrame, incumbents: pd.DataFrame, priors: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    incumbents = incumbents.drop(columns=["ID"])
    incumbent_df = df.merge(incumbents, left_on="ID", right_on="experiment_id")
    if priors is not None:
        priors = priors.drop(columns=["ID"])
        prior_df = df.merge(priors, left_on="ID", right_on="experiment_id")
    else:
        prior_df = None

    return incumbent_df, prior_df


def plot_subset(baseline_data: pd.DataFrame, dynabo_data: pd.DataFrame, pibo_data: pd.DataFrame, scenario_dataset: List[Tuple[str, str]]):
    fig, axs = plt.subplots(len(scenario_dataset), 1, figsize=(10, 10))
    for i, (scenario, dataset) in enumerate(scenario_dataset):
        plot_run_seaborn(baseline_data, dynabo_data, pibo_data, scenario, dataset, axs[i])
    plt.show()


def quantile_ci(data):
    return np.percentile(data, [2.5, 97.5])


def plot_run_seaborn(
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
    # Select relevant data
    relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_priors, relevant_pibo_incumbents, relevant_pibo_priors = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, prior_kind, use_rejection
    )

    full_range = pd.DataFrame({"after_n_evaluations": range(min_ntrials, max_ntrials + 1)})

    df_dict = {
        "baseline": relevant_baseline,
        "dynabo_incumbents": relevant_dynabo_incumbents,
        "pibo_incumbents": relevant_pibo_incumbents,
    }

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

    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["baseline"], label="baseline", ax=ax, errorbar=quantile_ci)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["dynabo_incumbents"], label="dynabo", ax=ax, errorbar=quantile_ci)
    sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df_dict["pibo_incumbents"], label="pibo", ax=ax, errorbar=quantile_ci)
    ax.set_ylabel("Regret")
    return ax


def get_priors(df: pd.DataFrame):
    df = df[df["performance"].notnull()]
    df = df[["experiment_id", "after_n_evaluations", "after_runtime", "after_virtual_runtime", "after_reasoning_runtime", "performance"]]

    percentile_bounds = list()

    for n_trials in sorted(df["after_n_evaluations"].unique()):
        relevant_df = df[df["after_n_evaluations"] == n_trials]
        after_runtime = relevant_df["after_runtime"].max()
        after_virtual_runtime = relevant_df["after_virtual_runtime"].max()
        after_reasoning_runtime = relevant_df["after_reasoning_runtime"].max()
        avg_prior_performance = relevant_df["performance"].dropna().mean()
        percentile_upper = np.percentile(relevant_df["performance"], 95)
        percentile_lower = np.percentile(relevant_df["performance"], 5)
        percentile_bounds.append([n_trials, after_runtime, after_virtual_runtime, after_reasoning_runtime, avg_prior_performance, percentile_upper, percentile_lower])

    percentile_bounds = pd.DataFrame(
        percentile_bounds, columns=["after_n_evaluations", "after_runtime", "after_virtual_runtime", "after_reasoning_runtime", "avg_prior_avg_performance", "percentile_upper", "percentile_lower"]
    )
    return percentile_bounds


def find_last(df: pd.DataFrame, experiment_id: int, column: str, current: int):
    last_trial = df[(df["experiment_id"] == experiment_id) & (df[column] < current)]
    if len(last_trial) == 0:
        raise ValueError("No previous trial found")
    else:
        last_column_value = df[(df["experiment_id"] == experiment_id) & (df[column] < current)][column].max()
        df = df[(df["experiment_id"] == experiment_id) & (df[column] == last_column_value)]
        return df


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
    if dataset is not None:
        relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario) & (baseline_data["dataset"] == dataset)]
        relevant_dynabo_incumbents = dynabo_incumbent_data[
            (dynabo_incumbent_data["scenario"] == scenario)
            & (dynabo_incumbent_data["dataset"] == dataset)
            & (dynabo_incumbent_data["prior_kind"] == prior_kind)
            & (dynabo_incumbent_data["validate_prior"] == use_rejection)
        ]
        relevant_dynabo_prior = dynabo_prior_data[
            (dynabo_prior_data["scenario"] == scenario)
            & (dynabo_prior_data["dataset"] == dataset)
            & (dynabo_prior_data["prior_kind"] == prior_kind)
            & (dynabo_prior_data["validate_prior"] == use_rejection)
        ]
        relevant_pibo_incumbents = pibo_incumbent_data[(pibo_incumbent_data["scenario"] == scenario) & (pibo_incumbent_data["dataset"] == dataset) & (pibo_incumbent_data["prior_kind"] == prior_kind)]
        relevant_pibo_prior = pibo_prior_data[
            (pibo_prior_data["scenario"] == scenario) & (pibo_prior_data["dataset"] == dataset) & (pibo_prior_data["prior_kind"] == prior_kind) & (pibo_prior_data["validate_prior"] == use_rejection)
        ]
    else:
        relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario)]
        relevant_dynabo_incumbents = dynabo_incumbent_data[
            (dynabo_incumbent_data["scenario"] == scenario) & (dynabo_incumbent_data["prior_kind"] == prior_kind) & (dynabo_incumbent_data["validate_prior"] == use_rejection)
        ]
        relevant_dynabo_prior = dynabo_prior_data[
            (dynabo_prior_data["scenario"] == scenario) & (dynabo_prior_data["prior_kind"] == prior_kind) & (dynabo_prior_data["validate_prior"] == use_rejection)
        ]
        relevant_pibo_incumbents = pibo_incumbent_data[(pibo_incumbent_data["scenario"] == scenario) & (pibo_incumbent_data["prior_kind"] == prior_kind)]
        relevant_pibo_prior = pibo_prior_data[(pibo_prior_data["scenario"] == scenario) & (pibo_prior_data["prior_kind"] == prior_kind) & (pibo_prior_data["validate_prior"] == use_rejection)]
    return relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_prior, relevant_pibo_incumbents, relevant_pibo_prior


def create_dataset_plots():
    for scenario in baseline_incumbent_df["scenario"].unique():
        scenario_df = baseline_incumbent_df[baseline_incumbent_df["scenario"] == scenario]
        for dataset in scenario_df["dataset"].unique():
            os.makedirs(f"plots/dataset_plots/regret/{scenario}", exist_ok=True)
            fig, axs = plt.subplots(3, 1, figsize=(30, 20))
            axs = axs.flatten()
            plot_number = 0
            for prior_kind in ["good", "medium", "misleading"]:
                ax = axs[plot_number]
                ax = plot_run_seaborn(
                    baseline_incumbent_df,
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
                )
                plot_number += 1
                ax.legend()
                ax.set_title(f"{prior_kind}")
            fig.suptitle(f"{scenario}")
            plt.savefig(
                f"plots/dataset_plots/regret/{scenario}/{dataset}.png",
                bbox_inches="tight",
            )
        plt.close()
        print(f"Saved {scenario}/{dataset}.png")


def create_scenario_plots():
    for scenario in baseline_incumbent_df["scenario"].unique():
        os.makedirs("plots/scenario_plots/regret", exist_ok=True)
        fig, axs = plt.subplots(3, 1, figsize=(30, 20))
        axs = axs.flatten()
        plot_number = 0
        for prior_kind in ["good", "medium", "misleading"]:
            ax = axs[plot_number]
            ax = plot_run_seaborn(
                baseline_incumbent_df,
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
            )
            plot_number += 1
            ax.legend()
            ax.set_title(f"{prior_kind}")
            fig.suptitle(f"{scenario}")
            plt.savefig(
                f"plots/scenario_plots/regret/{scenario}.png",
                bbox_inches="tight",
            )
        plt.close()
        print(f"Saved {scenario}/.png")


if __name__ == "__main__":
    gt_data = pd.read_csv("benchmark_data/gt_prior_data/origin_table.csv")
    gt_data = gt_data[gt_data["dataset"] != "CIFAR10"]
    gt_data = gt_data[gt_data["incumbent_trace"].notnull()]
    gt_data["incumbent_trace"] = gt_data["incumbent_trace"].apply(ast.literal_eval)
    gt_data["final_performance"] = gt_data["incumbent_trace"].apply(lambda x: x[-1][1])
    gt_data["dataset"] = gt_data["dataset"].astype(int)

    baseline_table = pd.read_csv(BASELINE_TABLE_PATH)
    baseline_incumbent_df = pd.read_csv(BASELINE_INCUMBENT_PATH)
    baseline_incumbent_df, _ = merge_df(baseline_table, baseline_incumbent_df, None)

    prior_table = pd.read_csv(PRIOR_TABLE_PATH)
    prior_incumbents = pd.read_csv(PRIOR_INCUMBENT_PATH)
    prior_priors = pd.read_csv(PRIOR_PRIORS_PATH)
    prior_incumbent_df, prior_prior_df = merge_df(prior_table, prior_incumbents, prior_priors)

    baseline_incumbent_df = baseline_incumbent_df.merge(gt_data, on=["scenario", "dataset"], suffixes=("", "_gt"))
    baseline_incumbent_df["regret"] = baseline_incumbent_df["final_performance_gt"] - baseline_incumbent_df["performance"]

    prior_incumbent_df = prior_incumbent_df.merge(gt_data, on=["scenario", "dataset"], suffixes=("", "_gt"))
    prior_incumbent_df["regret"] = prior_incumbent_df["final_performance_gt"] - prior_incumbent_df["performance"]

    prior_prior_df = prior_prior_df.merge(gt_data, on=["scenario", "dataset"], suffixes=("", "_gt"))
    prior_prior_df["regret"] = prior_prior_df["final_performance_gt"] - prior_prior_df["performance"]

    dynabo_incumbent_df = prior_incumbent_df[prior_incumbent_df["dynabo"] == True]
    dynabo_prior_df = prior_prior_df[prior_prior_df["dynabo"] == True]
    pibo_incumbent_df = prior_incumbent_df[prior_incumbent_df["pibo"] == True]
    pibo_prior_df = prior_prior_df[prior_prior_df["pibo"] == True]

    create_dataset_plots()
    create_scenario_plots()
# %%

# %%
