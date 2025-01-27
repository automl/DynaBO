# %%
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

baseline_table = pd.read_csv("dynabo/plots/baseline_table.csv")
baseline_incumbent = pd.read_csv("dynabo/plots/baseline_incumbent.csv")

dynabo_table = pd.read_csv("dynabo/plots/prior_table.csv")
dynabo_incumbent = pd.read_csv("dynabo/plots/prior_incumbent.csv")
dynabo_priors = pd.read_csv("dynabo/plots/prior_priors.csv")

pibo_table = pd.read_csv("dynabo/plots/pibo_table.csv")
pibo_incumbent = pd.read_csv("dynabo/plots/pibo_incumbent.csv")
pibo_priors = pd.read_csv("dynabo/plots/pibo_priors.csv")


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
        plot_run(baseline_data, dynabo_data, pibo_data, scenario, dataset, axs[i])
    plt.show()


def plot_run(
    baseline_data: pd.DataFrame,
    dynabo_incumbent_data: pd.DataFrame,
    dynabo_prior_data,
    pibo_incumbent_data: pd.DataFrame,
    pibo_prior_data,
    scenario: str,
    dataset: str,
    prior_kind: str,
    ax: plt.Axes,
    min_ntrials=1,
    max_ntrials=200,
):
    relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_priors, relevant_pibo_incumbents, relevant_pibo_priors = select_relevant_data(
        baseline_data, dynabo_incumbent_data, dynabo_prior_data, pibo_incumbent_data, pibo_prior_data, scenario, dataset, prior_kind
    )
    plot_baseline_incumbents_df = fill_df(relevant_baseline, min_ntrials)
    plot_dynabo_incubments_df = fill_df(relevant_dynabo_incumbents, min_ntrials)
    plot_pibo_incumbents_df = fill_df(relevant_pibo_incumbents, min_ntrials)

    dynabo_priors = get_priors(relevant_dynabo_priors)
    pibo_priors = get_priors(relevant_pibo_priors)

    ax.plot(plot_baseline_incumbents_df["after_n_evaluations"], plot_baseline_incumbents_df["avg_performance"], label="baseline")
    ax.fill_between(plot_baseline_incumbents_df["after_n_evaluations"], plot_baseline_incumbents_df["percentile_lower"], plot_baseline_incumbents_df["percentile_upper"], alpha=0.2)
    ax.plot(plot_dynabo_incubments_df["after_n_evaluations"], plot_dynabo_incubments_df["avg_performance"], label="dynabo")
    ax.fill_between(plot_dynabo_incubments_df["after_n_evaluations"], plot_dynabo_incubments_df["percentile_lower"], plot_dynabo_incubments_df["percentile_upper"], alpha=0.2)
    ax.plot(plot_pibo_incumbents_df["after_n_evaluations"], plot_pibo_incumbents_df["avg_performance"], label="pibo")
    ax.fill_between(plot_pibo_incumbents_df["after_n_evaluations"], plot_pibo_incumbents_df["percentile_lower"], plot_pibo_incumbents_df["percentile_upper"], alpha=0.2)

    ax.scatter(dynabo_priors["after_n_evaluations"], dynabo_priors["avg_prior_avg_performance"], label="dynabo_prior", color="red")
    ax.errorbar(
        dynabo_priors["after_n_evaluations"],
        dynabo_priors["avg_prior_avg_performance"],
        yerr=[
            dynabo_priors["avg_prior_avg_performance"] - dynabo_priors["percentile_lower"],
            dynabo_priors["percentile_upper"] - dynabo_priors["avg_prior_avg_performance"],
        ],
        fmt="none",  # Don't plot additional points
        ecolor="red",  # Color of the error bars
        capsize=3,  # Add small caps to error bars
        alpha=0.5,  # Transparency for aesthetics
    )
    ax.scatter(pibo_priors["after_n_evaluations"], pibo_priors["avg_prior_avg_performance"], label="pibo_prior", color="red")
    ax.errorbar(
        pibo_priors["after_n_evaluations"],
        pibo_priors["avg_prior_avg_performance"],
        yerr=[
            pibo_priors["avg_prior_avg_performance"] - pibo_priors["percentile_lower"],
            pibo_priors["percentile_upper"] - pibo_priors["avg_prior_avg_performance"],
        ],
        fmt="none",
        ecolor="blue",
        capsize=3,
        alpha=0.5,
    )

    return ax


def fill_df(iterator_df: pd.DataFrame, max_trials=200, x_axis_column: str = "after_n_evaluations"):
    rows = []
    for n_trials in sorted(iterator_df["after_n_evaluations"].unique()):
        if n_trials == 1:
            relevant_df = iterator_df[iterator_df["after_n_evaluations"] == n_trials]
            after_n_evaluations = n_trials
            after_runtime = relevant_df["after_runtime"].max()
            after_virtual_runtime = relevant_df["after_virtual_runtime"].max()
            after_reasoning_runtime = relevant_df["after_reasoning_runtime"].max()
            avg_performance = relevant_df["performance"].mean()
            std_performance = relevant_df["performance"].std()
            percentile_upper = np.percentile(relevant_df["performance"], 95)
            percentile_lower = np.percentile(relevant_df["performance"], 5)
            rows.append([after_n_evaluations, after_runtime, after_virtual_runtime, after_reasoning_runtime, avg_performance, std_performance, percentile_upper, percentile_lower])
        else:
            # Find row of last incumbent for each of the experiment_ids
            last_incumbent_rows = []
            for experiment_id in iterator_df["experiment_id"].unique():
                last = find_last(iterator_df, experiment_id, x_axis_column, n_trials)
                last_incumbent_rows.append(last)
            last_incumbent_df = pd.concat(last_incumbent_rows)
            after_n_evaluations = n_trials
            after_runtime = last_incumbent_df["after_runtime"].max()
            after_virtual_runtime = last_incumbent_df["after_virtual_runtime"].max()
            after_reasoning_runtime = last_incumbent_df["after_reasoning_runtime"].max()
            avg_performance = last_incumbent_df["performance"].mean()
            std_performance = last_incumbent_df["performance"].std()
            percentile_upper = np.percentile(last_incumbent_df["performance"], 95)
            percentile_lower = np.percentile(last_incumbent_df["performance"], 5)
            rows.append([after_n_evaluations, after_runtime, after_virtual_runtime, after_reasoning_runtime, avg_performance, std_performance, percentile_upper, percentile_lower])

    new_df = pd.DataFrame(
        rows, columns=["after_n_evaluations", "after_runtime", "after_virtual_runtime", "after_reasoning_runtime", "avg_performance", "std_performance", "percentile_upper", "percentile_lower"]
    )
    return new_df


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
    baseline_data: pd.DataFrame, dynabo_incumbent_data: pd.DataFrame, dynabo_prior_data, pibo_incumbent_data: pd.DataFrame, pibo_prior_data: pd.DataFrame, scenario: str, dataset: str, prior_kind
):
    relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario) & (baseline_data["dataset"] == dataset)]
    relevant_dynabo_incumbents = dynabo_incumbent_data[
        (dynabo_incumbent_data["scenario"] == scenario) & (dynabo_incumbent_data["dataset"] == dataset) & (dynabo_incumbent_data["prior_kind"] == prior_kind)
    ]
    relevant_dynabo_prior = dynabo_prior_data[(dynabo_prior_data["scenario"] == scenario) & (dynabo_prior_data["dataset"] == dataset) & (dynabo_prior_data["prior_kind"] == prior_kind)]
    relevant_pibo_incumbents = pibo_incumbent_data[(pibo_incumbent_data["scenario"] == scenario) & (pibo_incumbent_data["dataset"] == dataset) & (pibo_incumbent_data["prior_kind"] == prior_kind)]
    relevant_pibo_prior = pibo_prior_data[(pibo_prior_data["scenario"] == scenario) & (pibo_prior_data["dataset"] == dataset) & (pibo_prior_data["prior_kind"] == prior_kind)]
    return relevant_baseline, relevant_dynabo_incumbents, relevant_dynabo_prior, relevant_pibo_incumbents, relevant_pibo_prior


if __name__ == "__main__":
    baseline_df, _ = merge_df(baseline_table, baseline_incumbent, None)
    dynabo_df_incumbent_df, dynabo_prior_df = merge_df(dynabo_table, dynabo_incumbent, dynabo_priors)
    pibo_incumbent_df, pibo_prior_df = merge_df(pibo_table, pibo_incumbent, pibo_priors)

    baseline_df

    for dataset in baseline_df["dataset"].unique():
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs = axs.flatten()
        for ax, prior_kind in zip(axs, ["good", "medium", "misleading"]):
            try:
                ax = plot_run(baseline_df, dynabo_df_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df, "lcbench", dataset, prior_kind, ax, min_ntrials=1, max_ntrials=200)
            except Exception:
                pass
            ax.legend()
            ax.set_title(f"{prior_kind}")
        fig.suptitle(f"{dataset}")
        plt.savefig(
            f"dynabo/plots//lcbench/{dataset}.png",
            bbox_inches="tight",
        )
        plt.close()
