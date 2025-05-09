import os
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.data_processing.download_all_files import (
    YAHPO_DATA_GENERATION_INCUMBENT_MEDIUM_HARD_PATH,
    YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH,
    YAHPO_DATA_GENERATION_MEDIUM_HARD_PATH,
    YHAPO_DATA_GENERATION_ONE_SEED_PATH,
)


def load_data_one_seed():
    """
    Load the data from the csv files.
    """
    data_generation_table = pd.read_csv(YHAPO_DATA_GENERATION_ONE_SEED_PATH)
    data_generation_incumbent = pd.read_csv(YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH)
    data_generation_incumbent = data_generation_incumbent.drop(columns=["ID"])

    return data_generation_table, data_generation_incumbent


def load_data_multiple_seeds():
    """
    Load the data from the csv files.
    """
    data_generation_table = pd.read_csv(YAHPO_DATA_GENERATION_MEDIUM_HARD_PATH)
    data_generation_incumbent = pd.read_csv(YAHPO_DATA_GENERATION_INCUMBENT_MEDIUM_HARD_PATH)
    data_generation_incumbent = data_generation_incumbent.drop(columns=["ID"])

    return data_generation_table, data_generation_incumbent


def process_data_one_seed(
    data_generation_table: pd.DataFrame,
    data_generation_incumbents: pd.DataFrame,
    acquisition_function="expected_improvement",
    epsiolon=0.005,
):
    """
    Process the data and return the processed data.
    """
    # Select Final performace not NaN

    joined_data = pd.merge(data_generation_table, data_generation_incumbents, left_on="ID", right_on="experiment_id")
    joined_data = joined_data[joined_data["final_performance"].notna()]
    joined_data = joined_data[joined_data["acquisition_function"] == acquisition_function]

    joined_data["classification_threshold"] = joined_data["final_performance"] - joined_data["final_performance"] * epsiolon

    # Remove the incumbents
    joined_data = joined_data[joined_data["performance"] != joined_data["final_performance"]]

    joined_data = joined_data[joined_data["performance"] >= joined_data["classification_threshold"]]

    # For each ID select the lowest "n_trials" in the threshold
    joined_data = joined_data.sort_values(by=["ID", "n_trials"], ascending=[True, False])

    # Get first value for each ID while preserving ordering
    joined_data = joined_data.groupby("ID").first().reset_index()

    joined_data = joined_data.reset_index()

    joined_data = extract_difficulty(joined_data)

    joined_data = joined_data[["scenario", "dataset", "acquisition_function", "after_n_evaluations", "hard", "medium", "easy", "super_easy"]]
    return joined_data


def extract_min_difficulty(expected_improvement_data: pd.DataFrame, confidence_bound_data: pd.DataFrame):
    min_difficulty_data = deepcopy(expected_improvement_data)
    min_difficulty_data["acquisition_function"] = "min_difficulty"
    min_difficulty_data["after_n_evaluations"] = np.minimum(expected_improvement_data["after_n_evaluations"], confidence_bound_data["after_n_evaluations"])
    min_difficulty_data = extract_difficulty(min_difficulty_data)
    return min_difficulty_data


def extract_difficulty(data: pd.DataFrame):
    data["hard"] = data["after_n_evaluations"].apply(lambda x: x > 1000)
    data["medium"] = data["after_n_evaluations"].apply(lambda x: x > 500 and x <= 1000)
    data["easy"] = data["after_n_evaluations"].apply(lambda x: x > 200 and x <= 500)
    data["super_easy"] = data["after_n_evaluations"].apply(lambda x: x <= 200)

    return data


def process_data_multiple_seeds(data_generation_table: pd.DataFrame, data_generation_incumbents: pd.DataFrame, epsiolon: float = 0.005):
    """
    Process the data and return the processed data.
    """
    joined_data = pd.merge(data_generation_table, data_generation_incumbents, left_on="ID", right_on="experiment_id")
    joined_data = joined_data[joined_data["final_performance"].notna()]

    joined_data["classification_threshold"] = joined_data["final_performance"] - joined_data["final_performance"] * epsiolon

    # Remove the incumbents
    joined_data = joined_data[joined_data["performance"] != joined_data["final_performance"]]

    joined_data = joined_data[joined_data["performance"] >= joined_data["classification_threshold"]]

    # For each ID select the lowest "n_trials" in the threshold
    joined_data = joined_data.sort_values(by=["ID", "n_trials"], ascending=[True, False])

    # Get first value for each ID while preserving ordering
    joined_data = joined_data.groupby("ID").first().reset_index()

    joined_data = joined_data.reset_index()

    joined_data["hard"] = joined_data["after_n_evaluations"].apply(lambda x: x > 1000)
    joined_data["medium"] = joined_data["after_n_evaluations"].apply(lambda x: x > 500 and x <= 1000)
    joined_data["easy"] = joined_data["after_n_evaluations"].apply(lambda x: x > 200 and x <= 500)
    joined_data["super_easy"] = joined_data["after_n_evaluations"].apply(lambda x: x <= 200)

    joined_data = joined_data[["scenario", "dataset", "after_n_evaluations", "hard", "medium", "easy", "super_easy"]]
    return joined_data


def save_join_data(joined_data: pd.DataFrame, path):
    joined_data.to_csv(path, index=False)


def scenario_plots(joined_data: pd.DataFrame, prefix: str):
    # Ensure the output directory exists
    os.makedirs(prefix, exist_ok=True)

    scenarios = joined_data["scenario"].unique()

    for scenario in scenarios:
        # Create 4 subplots, only share y between violin plots (last 3)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        relevant_df = joined_data[joined_data["scenario"] == scenario]

        # Bar Plot of Difficulty
        difficulty_counts = relevant_df.groupby("acquisition_function")[["hard", "medium", "easy", "super_easy"]].sum()
        difficulty_counts = difficulty_counts.reset_index().melt(id_vars="acquisition_function", var_name="difficulty", value_name="value")
        sns.barplot(data=difficulty_counts, x="difficulty", y="value", ax=axs[0], hue="acquisition_function")
        axs[0].set_title("Difficulty Distribution")

        # Filter and plot violin plots with shared y-axis manually
        afs = ["expected_improvement", "confidence_bound", "min_difficulty"]
        titles = ["Expected Improvement", "Confidence Bound", "Min Difficulty"]

        # Determine global y-axis limits for violin plots
        violin_data = relevant_df[relevant_df["acquisition_function"].isin(afs)]
        y_min = violin_data["after_n_evaluations"].min()
        y_max = violin_data["after_n_evaluations"].max()

        for i, (af, title) in enumerate(zip(afs, titles), start=1):
            sns.violinplot(y="after_n_evaluations", data=relevant_df[relevant_df["acquisition_function"] == af], ax=axs[i])
            axs[i].set_title(title)
            axs[i].set_ylim(y_min, y_max)

        fig.suptitle(f"Scenario: {scenario}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{prefix}/{scenario}.png")
        plt.close(fig)


def dataset_plots(joined_data: pd.DataFrame, prefix: str):
    for scenario in joined_data["scenario"].unique():
        scenario_df = joined_data[joined_data["scenario"] == scenario]
        for dataset in scenario_df["dataset"].unique():
            dataset_df = scenario_df[scenario_df["dataset"] == dataset]

            fig, axs = plt.subplots(1, 2, figsize=(10, 20))
            # Bar Plot of Difficulty
            # Calculate how often hard medium ,...
            difficulty_counts = dataset_df[["hard", "medium", "easy", "super_easy"]].sum()
            difficulty_counts = difficulty_counts
            sns.barplot(x=difficulty_counts.index, y=difficulty_counts.values, ax=axs[0])

            # Violin plot for after_n_evaluations
            sns.violinplot(
                y="after_n_evaluations",
                data=dataset_df,
                ax=axs[1],
            )

            fig.suptitle(f"{scenario}_{dataset}")
            if not os.path.exists(f"{prefix}/{scenario}"):
                os.makedirs(f"{prefix}/{scenario}")
            plt.savefig(f"{prefix}/{scenario}/{dataset}.png")
            plt.close()


def one_seed():
    data_generation_table, data_generation_incumbent = load_data_one_seed()
    expected_imporvement_data = process_data_one_seed(data_generation_table, data_generation_incumbent, "expected_improvement")
    confidence_bound_data = process_data_one_seed(data_generation_table, data_generation_incumbent, "confidence_bound")
    min_difficulty_data = extract_min_difficulty(expected_imporvement_data, confidence_bound_data)
    joined_data = pd.concat([expected_imporvement_data, confidence_bound_data, min_difficulty_data])

    save_join_data(joined_data, "plotting_data/difficulty_groups_one_seed.csv")
    scenario_plots(joined_data, prefix="plots/metadata/one_seed/difficulty/")


def multiple_seeds():
    data_generation_table, data_generation_incumbent = load_data_multiple_seeds()
    joined_data = process_data_multiple_seeds(data_generation_table, data_generation_incumbent)
    save_join_data(joined_data, "plotting_data/difficulty_groups_multiple_seeds.csv")
    scenario_plots(
        joined_data,
        prefix="plots/metadata/multiple_seeds/difficulty/",
    )
    dataset_plots(
        joined_data,
        prefix="plots/metadata/multiple_seeds/difficulty/",
    )


if __name__ == "__main__":
    one_seed()
    # multiple_seeds()
