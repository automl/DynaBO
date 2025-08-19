import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from dynabo.utils.data_utils import extract_dataframe_from_column_and_after_n_evaluations, load_df, save_base_table


def execute_clustering(benchmark_name: str, table_name: str, only_incumbent: bool = False):
    scenario_dfs = prepare_data(benchmark_name, table_name, only_incumbent)
    for scenario, df in scenario_dfs.items():
        print(scenario)
        cluster_incumbents(df, scenario)


def prepare_data(benchmark_name: str, table_name: str, only_incumbent: bool = False):
    def extract_relevant_data(base_df: pd.DataFrame, scenario: str):
        scenario_df = base_df[base_df.scenario == scenario]
        costs = scenario_df[["cost"]].reset_index(drop=True)
        config_df = extract_dataframe_from_column_and_after_n_evaluations(scenario_df[["configuration"]])
        scenario_df = config_df.join(costs, how="left")
        return scenario_df

    save_base_table(benchmark_name, table_name, only_incumbent=only_incumbent)
    base_df = load_df(benchmark_name="mfpbench")

    scenario_dfs = {}
    for scenario in base_df.scenario.unique():
        scenario_dfs[scenario] = extract_relevant_data(base_df, scenario)

    return scenario_dfs


def cluster_incumbents(df: pd.DataFrame, scenario: str):
    # Use DBSCAN to cluster the data
    cluster_df = df.copy()
    cluster_df = cluster_df.drop(columns=["cost"])

    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(cluster_df)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)

    min_size = 10
    max_size = 300

    normalized_cost = (df["cost"] - df["cost"].min()) / (df["cost"].max() - df["cost"].min())
    sizes = (1 - normalized_cost) * (max_size - min_size) + min_size

    # --- Plotting Results ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Capture the scatter plot object
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        df["cost"] * -1,  # Using positive cost for a more intuitive Z-axis
        c=labels,
        cmap="viridis",
        s=sizes,
    )

    # 2. Create the legend from the scatter object
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    # Set titles and labels
    ax.set_title("DBSCAN Clustering with PCA")
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    ax.set_zlabel("Cost")

    plt.savefig(f"dbscan_clustering_pca_{scenario}.png")
    plt.close()
    print(f"Plot saved to dbscan_clustering_pca_{scenario}.png")


if __name__ == "__main__":
    execute_clustering(benchmark_name="mfpbench", table_name="data_generation_pd1", only_incumbent=True)
