# %%
import copy
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.data_processing.download_all_files import (
    BASELINE_INCUMBENT_PATH,
    BASELINE_TABLE_PATH,
    PRIOR_INCUMBENT_PATH,
    PRIOR_PRIORS_PATH,
    PRIOR_TABLE_PATH,
    PRIOR_WITH_DISREGARDING_INCUMBENT_PATH,
    PRIOR_WITH_DISREGARDING_PRIORS_PATH,
    PRIOR_WITH_DISREGARDING_TABLE_PATH,
)


def load_datageneration_data():
    main_table = pd.read_csv("plotting_data/datageneration_medium_hard.csv")
    configs = pd.read_csv("plotting_data/datageneration_incumbent_medium_hard.csv")

    main_table, _ = merge_df(main_table, configs, None)

    max_performances = get_max_performance([main_table])
    main_table = add_regret([main_table], max_performances)[0]
    return main_table


def load_performance_data():
    """
    Load the performance data, saved in the filesystem. Do some data_cleaning for lcbench and add regret.
    """

    def _clean_lcbench_performance(df: pd.DataFrame):
        mask = df["scenario"] == "lcbench"
        df.loc[mask, "performance"] = df.loc[mask, "performance"] / 100
        df.loc[mask, "final_performance"] = df.loc[mask, "final_performance"] / 100
        return df

    baseline_table = pd.read_csv(BASELINE_TABLE_PATH)
    baseline_config_df = pd.read_csv(BASELINE_INCUMBENT_PATH)
    baseline_config_df, _ = merge_df(baseline_table, baseline_config_df, None)

    prior_table_without_disregarding = pd.read_csv(PRIOR_TABLE_PATH)
    prior_configs_without_disregarding = pd.read_csv(PRIOR_INCUMBENT_PATH)
    prior_priors_without_disregarding = pd.read_csv(PRIOR_PRIORS_PATH)
    prior_config_without_disregarding, prior_prior_without_disregarding = merge_df(prior_table_without_disregarding, prior_configs_without_disregarding, prior_priors_without_disregarding)

    prior_table_with_disregarding = pd.read_csv(PRIOR_WITH_DISREGARDING_TABLE_PATH)
    prior_configs_with_disregarding = pd.read_csv(PRIOR_WITH_DISREGARDING_INCUMBENT_PATH)
    prior_priors_with_disregarding = pd.read_csv(PRIOR_WITH_DISREGARDING_PRIORS_PATH)
    prior_configs_with_disregarding, prior_priors_with_disregarding = merge_df(prior_table_with_disregarding, prior_configs_with_disregarding, prior_priors_with_disregarding)

    prior_config_df = pd.concat([prior_config_without_disregarding, prior_configs_with_disregarding])
    prior_priors_df = pd.concat([prior_prior_without_disregarding, prior_priors_with_disregarding])

    # For scenario lcbench divide by final_performance and perforamnce by 100
    baseline_config_df = _clean_lcbench_performance(baseline_config_df)
    prior_config_df = _clean_lcbench_performance(prior_config_df)
    prior_priors_df = _clean_lcbench_performance(prior_priors_df)

    max_performances = get_max_performance([baseline_config_df, prior_config_df])
    baseline_config_df, prior_config_without_disregarding, prior_prior_without_disregarding = add_regret([baseline_config_df, prior_config_df, prior_priors_df], max_performances)
    return baseline_config_df, prior_config_df, prior_priors_df


def merge_df(df: pd.DataFrame, incumbents: pd.DataFrame, priors: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """ "
    Merge different logging dataframes on the id column
    """
    incumbents = incumbents.drop(columns=["ID"])
    incumbent_df = df.merge(incumbents, left_on="ID", right_on="experiment_id")
    if priors is not None:
        priors = priors.drop(columns=["ID"])
        prior_df = df.merge(priors, left_on="ID", right_on="experiment_id")
    else:
        prior_df = None

    return incumbent_df, prior_df


def get_max_performance(dfs: Tuple[pd.DataFrame]) -> Dict[Tuple[str, int], float]:
    """
    Compute the maximum performance.
    """
    concat_df = pd.concat(dfs)
    max_performances = concat_df.groupby(["scenario", "dataset"])["performance"].max()
    return max_performances.to_dict()


def add_regret(dfs: List[pd.DataFrame], max_performances: Dict[Tuple[str, int], float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the regret to the dataframes.
    """
    for df in dfs:
        max_perf_series = pd.Series(max_performances)
        # Use the scenario and dataset columns to index the Series—
        keys = pd.MultiIndex.from_arrays([df["scenario"], df["dataset"]])

        # Retrieve the corresponding max values for each row
        max_values = max_perf_series.loc[keys].values

        # Compute regret values
        df["regret"] = max_values - df["performance"].values
        df["final_regret"] = max_values - df["final_performance"].values
    return dfs


def split_dynabo_and_pibo(prior_config_df: pd.DataFrame, prior_prior_df: pd.DataFrame):
    dynabo_incumbent_df = prior_config_df[prior_config_df["dynabo"] == True]
    dynabo_prior_df = prior_prior_df[prior_prior_df["dynabo"] == True]
    pibo_incumbent_df = prior_config_df[prior_config_df["pibo"] == True]
    pibo_prior_df = prior_prior_df[prior_prior_df["pibo"] == True]

    return dynabo_incumbent_df, dynabo_prior_df, pibo_incumbent_df, pibo_prior_df


def filter_prior_approach(
    incumbent_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    select_dynabo: bool,
    select_pibo: bool,
    with_validating: bool,
    prior_validation_method: str,
    prior_validation_manwhitney_p: Optional[float],
    prior_validation_difference_threshold: Optional[float] = None,
):
    assert select_dynabo ^ select_pibo
    incumbent_df = incumbent_df[(incumbent_df["dynabo"] == select_dynabo) & (incumbent_df["pibo"] == select_pibo) & (incumbent_df["validate_prior"] == with_validating)]
    prior_df = prior_df[(prior_df["dynabo"] == select_dynabo) & (prior_df["pibo"] == select_pibo) & (prior_df["validate_prior"] == with_validating)]
    if prior_validation_method == "mann_whitney_u":
        incumbent_df = incumbent_df[incumbent_df["prior_validation_method"] == "mann_whitney_u"]
        prior_df = prior_df[prior_df["prior_validation_method"] == "mann_whitney_u"]

        incumbent_df = incumbent_df[incumbent_df["prior_validation_manwhitney_p"] == prior_validation_manwhitney_p]
        prior_df = prior_df[prior_df["prior_validation_manwhitney_p"] == prior_validation_manwhitney_p]

    elif prior_validation_method == "difference":
        incumbent_df = incumbent_df[incumbent_df["prior_validation_method"] == "difference"]
        prior_df = prior_df[prior_df["prior_validation_method"] == "difference"]
        incumbent_df = incumbent_df[incumbent_df["prior_validation_difference_threshold"] == prior_validation_difference_threshold]
        prior_df = prior_df[prior_df["prior_validation_difference_threshold"] == prior_validation_difference_threshold]

    return incumbent_df, prior_df


def plot_final_run(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    scenario: str,
    dataset: str,
    prior_kind: str,
    ax: plt.Axes,
    min_ntrials=1,
    max_ntrials=200,
    error_bar_type: str = "se",
):
    config_dict = copy.deepcopy(config_dict)
    prior_dict = copy.deepcopy(prior_dict)
    # Select relevant data
    config_dict, prior_dict = preprocess_configs(
        config_dict,
        prior_dict,
        scenario,
        dataset,
        prior_kind,
        min_ntrials,
        max_ntrials,
    )

    for key, df in config_dict.items():
        sns.lineplot(x="after_n_evaluations", y="regret", drawstyle="steps-pre", data=df, label=key, ax=ax, errorbar=error_bar_type)

    # Check highest performacne after 10 trials
    highest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == 40]["regret"].mean()
    smallest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == 200]["regret"].mean()

    # TODO this is the issue
    ax.set_ylim(smallest_regret * 0.1, highest_regret * 1.1)
    ax.set_ylabel("Regret")

    return ax


def preprocess_configs(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    scenario: str,
    dataset: str,
    prior_kind: str,
    min_ntrials=1,
    max_ntrials=200,
):
    config_dict, prior_dict = select_relevant_data(
        config_dict,
        prior_dict,
        scenario,
        dataset,
        prior_kind,
    )

    config_dict = extract_incumbent_steps(df_dict=config_dict, min_ntrials=min_ntrials, max_ntrials=max_ntrials)

    return config_dict, prior_dict


def select_relevant_data(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    scenario: str,
    dataset: str,
    prior_kind: str,
):
    # Select configuraitons that should be plotted
    for key in config_dict.keys():
        df = config_dict[key]
        # only consider incumbent
        df = df[(df["incumbent"] == 1)]

        # If priors used only select the relevant prior
        if any(df["dynabo"] == True) or any(df["pibo"] == True):
            df = df[df["prior_kind"] == prior_kind]
        config_dict[key] = df

    # Select the priors that sohuld be plotted
    for key in prior_dict.keys():
        df = prior_dict[key]
        # If priors used only select the relevant prior
        if any(df["dynabo"] == True) or any(df["pibo"] == True):
            df = df[(df["prior_kind"] == prior_kind)]
        prior_dict[key] = df

    if scenario is not None:  # Select relevant based on scenario
        config_dict = {key: df[(df["scenario"] == scenario)] for key, df in config_dict.items()}
        prior_dict = {key: df[(df["scenario"] == scenario)] for key, df in prior_dict.items()}

    if scenario is not None and dataset is not None:  # select relevant based on dataset
        config_dict = {key: df[(df["dataset"] == dataset)] for key, df in config_dict.items()}
        prior_dict = {key: df[(df["dataset"] == dataset)] for key, df in prior_dict.items()}

    return config_dict, prior_dict


def extract_incumbent_steps(df_dict: Dict[str, pd.DataFrame], min_ntrials: int, max_ntrials: int):
    full_range = pd.DataFrame({"after_n_evaluations": range(min_ntrials, max_ntrials + 1)})

    # Step 1: Iterate over all DataFrames
    for key, df in df_dict.items():
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
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    error_bar_type: str,
    scenarios: List[str],
):
    for scenario in scenarios:
        datasets = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["scenario"] == scenario]["dataset"].unique()
        for dataset in datasets:
            os.makedirs("plots/dataset_plots/regret/scenario", exist_ok=True)
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)  # Wider and higher resolution
            axs = axs.flatten()
            plot_number = 0
            for prior_kind in ["good", "medium", "misleading"]:
                ax = axs[plot_number]
                ax = plot_final_run(
                    config_dict,
                    prior_dict,
                    scenario,
                    dataset,
                    prior_kind,
                    ax=ax,
                    min_ntrials=1,
                    max_ntrials=200,
                    error_bar_type=error_bar_type,
                )
                plot_number += 1
                set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")
            set_fig_style(fig, axs, f"Average regret on {scenario}")
            save_fig(f"plots/dataset_plots/regret/scenario/{dataset}.pdf")
            print(f"Saved {scenario}")


def create_scenario_plots(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    error_bar_type: str,
    scenarios: List[str],
):
    for scenario in scenarios:
        os.makedirs("plots/scenario_plots/regret", exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)  # Wider and higher resolution
        axs = axs.flatten()
        plot_number = 0
        for prior_kind in ["good", "medium", "misleading"]:
            ax = axs[plot_number]
            ax = plot_final_run(
                config_dict,
                prior_dict,
                scenario,
                None,
                prior_kind,
                ax=ax,
                min_ntrials=1,
                max_ntrials=200,
                error_bar_type=error_bar_type,
            )
            plot_number += 1
            set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")
        set_fig_style(fig, axs, f"Average regret on {scenario}")
        save_fig(f"plots/scenario_plots/regret/{scenario}.pdf")
        print(f"Saved {scenario}")


def create_overall_plot(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    error_bar_type: str,
):
    os.makedirs("plots/scenario_plots/regret", exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    axs = axs.flatten()

    plot_number = 0
    for prior_kind in ["good", "medium", "misleading"]:
        ax = axs[plot_number]

        # Call the plotting function
        ax = plot_final_run(
            config_dict,
            prior_dict,
            None,
            None,
            prior_kind,
            ax=ax,
            min_ntrials=1,
            max_ntrials=200,
            error_bar_type=error_bar_type,
        )

        set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")

        plot_number += 1

    set_fig_style(fig, axs, "Overall Regret Across Different Priors")

    save_fig("plots/scenario_plots/regret/overall.png")


def set_ax_style(ax, prior_kind: str, x_label, y_label):
    # Remove ax legend
    ax.legend().remove()

    if prior_kind == "good":
        prior_name = "Informative"
    elif prior_kind == "medium":
        prior_name = "Mixed"
    elif prior_kind == "misleading":
        prior_name = "Misleading"

    # Improve title aesthetics
    ax.set_title(
        f"Prior: {prior_name}",
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
    baseline_config_df, prior_config_df, prior_prior_df = load_performance_data()
    dynabo_incumbent_df_with_validation_05, dynabo_prior_df_with_validation_05 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=True,
        prior_validation_method="mann_whitney_u",
        prior_validation_manwhitney_p=0.05,
        prior_validation_difference_threshold=None,
    )
    dynabo_incumbent_df_with_validation_difference_05, dynabo_prior_df_with_validation_difference_05 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-0.5,
    )
    dynabo_incumbent_df_with_validation_difference_1, dynabo_prior_df_with_validation_difference_1 = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=True,
        prior_validation_method="difference",
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=-1,
    )

    dynabo_incumbent_df_without_validation, dynabo_prior_df_without_validation = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=True,
        select_pibo=False,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=0.05,  # TODO change this to none after downloading data again
        prior_validation_difference_threshold=None,
    )
    pibo_incumbent_df_without_validation, pibo_prior_df_without_validation = filter_prior_approach(
        incumbent_df=prior_config_df,
        prior_df=prior_prior_df,
        select_dynabo=False,
        select_pibo=True,
        with_validating=False,
        prior_validation_method=None,
        prior_validation_manwhitney_p=None,
        prior_validation_difference_threshold=None,
    )

    config_dict = {
        "Vanilla BO": baseline_config_df,
        "DynaBO, p=None": dynabo_incumbent_df_without_validation,
        "DynaBO, MWU p=0.05": dynabo_incumbent_df_with_validation_05,
        "DynaBO, difference=-1": dynabo_incumbent_df_with_validation_difference_1,
        "PiBO": pibo_incumbent_df_without_validation,
    }

    prior_dict = {
        "DynaBO, p=None": dynabo_prior_df_without_validation,
        "DynaBO, MWU p=0.05": dynabo_prior_df_with_validation_05,
        "DynaBO, difference=-1": dynabo_prior_df_with_validation_difference_1,
        "PiBO": pibo_prior_df_without_validation,
    }
    # create_dataset_plots(
    #    config_dict=config_dict,
    #    prior_dict=prior_dict,
    #    error_bar_type="se",
    #    scenarios=baseline_config_df["scenario"].unique(),
    # )

    create_scenario_plots(
        config_dict,
        prior_dict,
        error_bar_type="se",
        scenarios=baseline_config_df["scenario"].unique(),
    )
    create_overall_plot(
        config_dict,
        prior_dict,
        error_bar_type="se",
    )


if __name__ == "__main__":
    plot_final_results()
    # plot_datageneration()
