import copy
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_datageneration_data():
    main_table = pd.read_csv("plotting_data/datageneration_medium_hard.csv")
    configs = pd.read_csv("plotting_data/datageneration_incumbent_medium_hard.csv")

    main_table, _ = merge_df(main_table, configs, None)

    min_costs = get_min_costs([main_table])
    main_table = add_regret([main_table], min_costs)[0]
    return main_table


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


def get_min_costs(dfs: Tuple[pd.DataFrame], benchmarklib: str) -> Dict[Tuple[str, int], float]:
    """
    Compute the minimum cost.
    """
    concat_df = pd.concat(dfs)
    if benchmarklib == "yahpogym":
        index = ["scenario", "dataset"]
    elif benchmarklib == "mfpbench":
        index = ["scenario"]
    min_costs = concat_df.groupby(index)["cost"].min()
    return min_costs.to_dict()


def add_regret(dfs: List[pd.DataFrame], min_costs: Dict[Tuple[str, int], float], benchmarklib: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the regret to the dataframes.
    """
    for df in dfs:
        min_cost_series = pd.Series(min_costs)
        # Use the scenario and dataset columns to index the Series—
        if benchmarklib == "yahpogym":
            keys = pd.MultiIndex.from_arrays([df["scenario"], df["dataset"]])
        elif benchmarklib == "mfpbench":
            keys = df["scenario"]

        min_cost_values = min_cost_series.loc[keys].values
        df["regret"] = df["cost"].values - min_cost_values
        df["final_regret"] = df["final_cost"].values - min_cost_values

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
    prior_std_denominator: int,
    prior_decay_enumerator: int,
    prior_static_position: Optional[bool],
    prior_every_n_trials: Optional[int],
    validate_prior: Optional[bool],
    prior_validation_method: Optional[str],
    prior_validation_manwhitney_p: Optional[float],
    prior_validation_difference_threshold: Optional[float] = None,
):
    assert select_dynabo ^ select_pibo

    incumbent_df = incumbent_df[incumbent_df["prior_decay_enumerator"] == prior_decay_enumerator]
    incumbent_df = incumbent_df[incumbent_df["prior_std_denominator"] == prior_std_denominator]

    if select_dynabo:
        incumbent_df = incumbent_df[incumbent_df["dynabo"] == True]
        prior_df = prior_df[prior_df["dynabo"] == True]

        if prior_static_position:
            incumbent_df = incumbent_df[(incumbent_df["prior_static_position"] == True) & (incumbent_df["prior_every_n_trials"] == prior_every_n_trials)]
            prior_df = prior_df[(prior_df["prior_static_position"] == True) & (prior_df["prior_every_n_trials"] == prior_every_n_trials)]
        else:
            raise NotImplementedError("No plotting implemented for non-static position")

        if validate_prior:
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

            elif prior_validation_method == "baseline_perfect":
                incumbent_df = incumbent_df[incumbent_df["prior_validation_method"] == "baseline_perfect"]
                prior_df = prior_df[prior_df["prior_validation_method"] == "baseline_perfect"]

        else:
            incumbent_df = incumbent_df[incumbent_df["validate_prior"] == False]
            prior_df = prior_df[prior_df["validate_prior"] == False]
    else:  # select pibo
        incumbent_df = incumbent_df[incumbent_df["pibo"] == True]
        prior_df = prior_df[prior_df["pibo"] == True]

    return incumbent_df, prior_df


def plot_final_run(
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
    config_dict = copy.deepcopy(config_dict)
    prior_dict = copy.deepcopy(prior_dict)
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

    for key, df in config_dict.items():
        sns.lineplot(
            x="after_n_evaluations",
            y="regret",
            drawstyle="steps-pre",
            data=df,
            label=key,
            ax=ax,
            errorbar=error_bar_type,
            color=style_dict[key]["color"],
            linestyle=style_dict[key]["linestyle"],
            marker=style_dict[key]["marker"],
            markersize=8,
            markevery=5,
        )

    if benchmarklib == "yahpogym":
        # check smallest regret after 50 trials
        highest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == (50)]["regret"].mean()
        smallest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == max_ntrials]["regret"].mean()
        ax.set_ylim(smallest_regret * 0.1, highest_regret * 1.1)
    elif benchmarklib == "mfpbench":
        highest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == (max_ntrials - 20)]["regret"].mean()
        smallest_regret = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["after_n_evaluations"] == max_ntrials]["regret"].mean()
        ax.set_ylim(smallest_regret * 0.1, highest_regret * 1.5)

    return ax


def preprocess_configs(
    config_dict: Dict[str, pd.DataFrame],
    prior_dict: Dict[str, pd.DataFrame],
    scenario: str,
    dataset: str,
    prior_kind: str,
    benchmarklib: str,
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

    config_dict = extract_incumbent_steps(df_dict=config_dict, min_ntrials=min_ntrials, max_ntrials=max_ntrials, benchmarklib=benchmarklib)

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


def extract_incumbent_steps(df_dict: Dict[str, pd.DataFrame], min_ntrials: int, max_ntrials: int, benchmarklib: str):
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
            if benchmarklib == "yahpogym":
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
            elif benchmarklib == "mfpbench":
                # Step 4: Iterate over each seed in the current dataset
                for seed in scenario_df["seed"].unique():
                    seed_mask = scenario_mask & (df["seed"] == seed)

                    # Merge the full range with the current group to ensure all `after_n_evaluations` are included
                    merged_df = full_range.merge(df.loc[seed_mask], on="after_n_evaluations", how="left")

                    merged_df["regret"] = merged_df["regret"].ffill()
                    merged_df_final = pd.DataFrame(columns=["scenario", "dataset", "seed", "after_n_evaluations", "regret"])
                    merged_df_final["after_n_evaluations"] = merged_df["after_n_evaluations"]
                    merged_df_final["regret"] = merged_df["regret"]
                    merged_df_final["scenario"] = scenario
                    merged_df_final["dataset"] = None
                    merged_df_final["seed"] = seed

                    local.append(merged_df_final)

        # Concatenate the local list to a DataFrame
        df_dict[key] = pd.concat(local)
    return df_dict


def set_fig_style(fig, axs, title: str, ncol):
    fig.suptitle(title, fontsize=25, fontweight="bold", y=1)

    # Extract all plotted lines from axs and only keep unique lines
    label_line_dict = {line.get_label(): line for ax in axs for line in ax.get_lines()}
    labels, lines = zip(*label_line_dict.items())

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
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


def create_dataset_plots(config_dict: Dict[str, pd.DataFrame], prior_dict: Dict[str, pd.DataFrame], error_bar_type: str, scenarios: List[str], benchmarklib: str, base_path: str, ncol: int):
    if benchmarklib == "yahpogym":
        min_ntrials = 1
        max_n_trials = 200
    elif benchmarklib == "mfpbench":
        min_ntrials = 1
        max_n_trials = 50

    for scenario in scenarios:
        datasets = config_dict["Vanilla BO"][config_dict["Vanilla BO"]["scenario"] == scenario]["dataset"].unique()
        for dataset in datasets:
            os.makedirs(f"{base_path}/{scenario}", exist_ok=True)
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
                    benchmarklib=benchmarklib,
                    min_ntrials=min_ntrials,
                    max_ntrials=max_n_trials,
                    error_bar_type=error_bar_type,
                )
                plot_number += 1
                set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")
            set_fig_style(fig, axs, f"Average regret on {scenario}", ncol=ncol)
            save_fig(f"{base_path}/{scenario}/{dataset}.pdf")
        print(f"Saved {scenario}")


def create_scenario_plots(
    config_dict: Dict[str, pd.DataFrame], prior_dict: Dict[str, pd.DataFrame], style_dict: Dict[str, str], error_bar_type: str, scenarios: List[str], benchmarklib: str, base_path: str, ncol: int
):
    if benchmarklib == "yahpogym":
        min_ntrials = 1
        max_n_trials = 200
    elif benchmarklib == "mfpbench":
        min_ntrials = 1
        max_n_trials = 50
    for scenario in scenarios:
        os.makedirs(f"plots/scenario_plots/{benchmarklib}//regret", exist_ok=True)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6), dpi=300)  # Wider and higher resolution
        axs = axs.flatten()
        plot_number = 0
        for prior_kind in ["good", "medium", "misleading", "deceiving"]:
            ax = axs[plot_number]
            ax = plot_final_run(
                config_dict,
                prior_dict,
                style_dict,
                scenario,
                None,
                prior_kind,
                ax=ax,
                benchmarklib=benchmarklib,
                min_ntrials=min_ntrials,
                max_ntrials=max_n_trials,
                error_bar_type=error_bar_type,
            )
            plot_number += 1
            set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")
        set_fig_style(fig, axs, f"Average regret on {scenario}", ncol=ncol)
        save_fig(f"{base_path}/{benchmarklib}/regret/{scenario}.pdf")
        print(f"Saved {scenario}")


def create_overall_plot(
    config_dict: Dict[str, pd.DataFrame], prior_dict: Dict[str, pd.DataFrame], style_dict: Dict[str, Dict[str, str]], error_bar_type: str, benchnmarklib: str, base_path: str, ncol: int
):
    os.makedirs(f"plots/scenario_plots/{benchnmarklib}/regret", exist_ok=True)
    if benchnmarklib == "yahpogym":
        min_ntrials = 1
        max_n_trials = 200
    elif benchnmarklib == "mfpbench":
        min_ntrials = 1
        max_n_trials = 50
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), dpi=300)
    axs = axs.flatten()

    plot_number = 0
    for prior_kind in ["good", "medium", "misleading", "deceiving"]:
        ax = axs[plot_number]

        # Call the plotting function
        ax = plot_final_run(
            config_dict,
            prior_dict,
            style_dict,
            None,
            None,
            prior_kind,
            ax=ax,
            benchmarklib=benchnmarklib,
            min_ntrials=min_ntrials,
            max_ntrials=max_n_trials,
            error_bar_type=error_bar_type,
        )

        set_ax_style(ax, prior_kind=prior_kind, x_label="Number of Evaluations", y_label="Regret")

        plot_number += 1

    set_fig_style(fig, axs, "Overall Regret Across Different Priors", ncol=ncol)

    save_fig(f"{base_path}/{benchnmarklib}/regret/overall.pdf")


def set_ax_style(ax, prior_kind: str, x_label, y_label):
    # Remove ax legend
    ax.legend().remove()

    if prior_kind == "good":
        prior_name = "Informative"
    elif prior_kind == "medium":
        prior_name = "Semi-Informative"
    elif prior_kind == "misleading":
        prior_name = "Misleading"
    elif prior_kind == "deceiving":
        prior_name = "Deceiving"
    elif prior_kind == "dummy_value":
        prior_name = "Prior Accept"
    else:
        raise ValueError(f"Prior kind {prior_kind} not supported")
    # Improve title aesthetics
    ax.set_title(
        f"{prior_name}",
        fontsize=30,
        fontweight="bold",
    )

    # Enhance axes labels and grid
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlabel(x_label, fontsize=25, fontweight="bold")
    if prior_kind == "good":
        # remove y axis
        ax.set_ylabel(y_label, fontsize=25, fontweight="bold")
    else:
        ax.set_ylabel(None)
        ax.set_yticklabels([])
