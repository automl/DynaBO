import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from dynabo.plotting.download_all_files import DATA_GENERATION_INCUMBENT_PATH, DATA_GENERATION_TABLE_PATH


def load_data():
    """
    Load the data from the csv files.
    """
    data_generation_table = pd.read_csv(DATA_GENERATION_TABLE_PATH)
    data_generation_incumbent = pd.read_csv(DATA_GENERATION_INCUMBENT_PATH)
    data_generation_incumbent = data_generation_incumbent.drop(columns=["ID"])

    return data_generation_table, data_generation_incumbent


def process_data(data_generation_table, data_generation_incumbents, epsiolon=0.005):
    """
    Process the data and return the processed data.
    """
    # Select Final performace not NaN

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


def scenario_plots(joined_data: pd.DataFrame):
    # Scenario Plots
    scenarios = joined_data["scenario"].unique()

    for scenario in scenarios:
        fig, axs = plt.subplots(1, 2, figsize=(10, 20))
        relevant_df = joined_data[joined_data["scenario"] == scenario]

        # Bar Plot of Difficulty
        # Calculate how often hard medium ,...
        difficulty_counts = relevant_df[["hard", "medium", "easy", "super_easy"]].sum()
        difficulty_counts = difficulty_counts
        sns.barplot(x=difficulty_counts.index, y=difficulty_counts.values, ax=axs[0])

        # Violin plot for after_n_evaluations
        sns.violinplot(
            y="after_n_evaluations",
            data=relevant_df,
            ax=axs[1],
        )

        fig.suptitle(scenario)
        plt.savefig(f"plots/metadata/difficulty/{scenario}.pdf")
        plt.close()

    # Total Plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))
    # Bar Plot of Difficulty
    # Calculate how often hard medium ,...
    difficulty_counts = joined_data[["hard", "medium", "easy", "super_easy"]].sum()
    difficulty_counts = difficulty_counts
    sns.barplot(x=difficulty_counts.index, y=difficulty_counts.values, ax=axs[0])

    # Violin plot for after_n_evaluations
    sns.violinplot(
        y="after_n_evaluations",
        data=joined_data,
        ax=axs[1],
    )

    fig.suptitle("Total")
    plt.savefig("plots/metadata/difficulty/total.pdf")


if __name__ == "__main__":
    data_generation_table, data_generation_incumbent = load_data()
    joined_data = process_data(data_generation_table, data_generation_incumbent)
    save_join_data(joined_data, "plotting_data/difficulty_groups.csv")
    scenario_plots(joined_data)
