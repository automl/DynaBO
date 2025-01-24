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


def merge_df(df: pd.DataFrame, incumbents: pd.DataFrame, priors: Optional[pd.DataFrame]) -> pd.DataFrame:
    incumbents = incumbents.drop(columns=["ID"])
    if priors is not None:
        priors = priors.drop(columns=["ID"])
        priors = priors[["experiment_id", "after_n_evaluations", "performance"]]
        priors.columns = ["experiment_id", "after_n_evaluations", "prior_performance"]

    df = df.merge(incumbents, left_on="ID", right_on="experiment_id")
    if priors is not None:
        df = df.merge(priors, on=["experiment_id", "after_n_evaluations"], how="left")

    return df


def plot_subset(baseline_data: pd.DataFrame, dynabo_data: pd.DataFrame, pibo_data: pd.DataFrame, scenario_dataset: List[Tuple[str, str]]):
    fig, axs = plt.subplots(len(scenario_dataset), 1, figsize=(10, 10))
    for i, (scenario, dataset) in enumerate(scenario_dataset):
        plot_run(baseline_data, dynabo_data, pibo_data, scenario, dataset, axs[i])
    plt.show()


def plot_run(baseline_data: pd.DataFrame, dynabo_data: pd.DataFrame, pibo_data: pd.DataFrame, scenario: str, dataset: str, prior_kind: str, ax: plt.Axes, min_ntrials=1, max_ntrials=200):
    relevant_baseline, relevant_dynabo, relevant_pibo = select_relevant_data(baseline_data, dynabo_data, pibo_data, scenario, dataset, prior_kind)
    relevant_baseline = fill_df(relevant_baseline, min_ntrials)
    relevant_dynabo = fill_df(relevant_dynabo, min_ntrials)
    relevant_pibo = fill_df(relevant_pibo, min_ntrials)

    ax.plot(relevant_baseline["after_n_evaluations"], relevant_baseline["avg_performance"], label="baseline")
    ax.fill_between(relevant_baseline["after_n_evaluations"], relevant_baseline["percentile_lower"], relevant_baseline["percentile_upper"], alpha=0.2)
    ax.plot(relevant_dynabo["after_n_evaluations"], relevant_dynabo["avg_performance"], label="dynabo")
    ax.fill_between(relevant_dynabo["after_n_evaluations"], relevant_dynabo["percentile_lower"], relevant_dynabo["percentile_upper"], alpha=0.2)
    ax.plot(relevant_pibo["after_n_evaluations"], relevant_pibo["avg_performance"], label="pibo")
    ax.fill_between(relevant_pibo["after_n_evaluations"], relevant_pibo["percentile_lower"], relevant_pibo["percentile_upper"], alpha=0.2)
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


def find_last(df: pd.DataFrame, experiment_id: int, column: str, current: int):
    last_trial = df[(df["experiment_id"] == experiment_id) & (df[column] < current)]
    if len(last_trial) == 0:
        raise ValueError("No previous trial found")
    else:
        last_column_value = df[(df["experiment_id"] == experiment_id) & (df[column] < current)][column].max()
        df = df[(df["experiment_id"] == experiment_id) & (df[column] == last_column_value)]
        return df


def select_relevant_data(baseline_data: pd.DataFrame, dynabo_data: pd.DataFrame, pibo_data: pd.DataFrame, scenario: str, dataset: str, prior_kind):
    relevant_baseline = baseline_data[(baseline_data["scenario"] == scenario) & (baseline_data["dataset"] == dataset)]
    relevant_dynabo = dynabo_data[(dynabo_data["scenario"] == scenario) & (dynabo_data["dataset"] == dataset) & (dynabo_data["prior_kind"] == prior_kind)]
    relevant_pibo = pibo_data[(pibo_data["scenario"] == scenario) & (pibo_data["dataset"] == dataset) & (pibo_data["prior_kind"] == prior_kind)]
    return relevant_baseline, relevant_dynabo, relevant_pibo


baseline_df = merge_df(baseline_table, baseline_incumbent, None)
dynabo_df = merge_df(dynabo_table, dynabo_incumbent, dynabo_priors)
pibo_df = merge_df(pibo_table, pibo_incumbent, pibo_priors)

baseline_df

for dataset in baseline_df["dataset"].unique():
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs = axs.flatten()
    for ax, prior_kind in zip(axs, ["good", "medium", "misleading"]):
        try:
            ax = plot_run(baseline_df, dynabo_df, pibo_df, "lcbench", dataset, prior_kind, ax, min_ntrials=1, max_ntrials=200)
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
