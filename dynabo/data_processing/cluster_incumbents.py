import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from gower import gower_matrix

from dynabo.utils.data_utils import extract_dataframe_from_column_and_after_n_evaluations, load_df, save_base_table


def execute_clustering(benchmark_name: str, table_name: str, only_incumbent: bool, use_pca: bool ):
    scenario_dfs = prepare_data(benchmark_name, table_name, only_incumbent)
    for scenario, df in scenario_dfs.items():
        print(scenario)
        cluster_incumbents(df, scenario, use_pca=use_pca)


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


def cluster_incumbents(df: pd.DataFrame, scenario: str, use_pca: bool):
    # Use DBSCAN to cluster the data
    cluster_df = df.copy()
    cluster_df = cluster_df.drop(columns=["cost"])

    distance_matrix = gower_matrix(cluster_df) 
    
    agg = AgglomerativeClustering(
        n_clusters=20,
        linkage="average"
    )
    labels_agg = agg.fit_predict(distance_matrix)

    if use_pca:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df)
    else:
        df = df.drop(columns=["config_lr_power" , "config_opt_momentum"])
        reduced_data = df.to_numpy()

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
        c=labels_agg,
        cmap="viridis",
        s=sizes,
    )

    # 2. Create the legend from the scatter object
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    # Set titles and labels
    ax.set_title("DBSCAN Clustering")
    if use_pca:
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    else:
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
    ax.set_zlabel("Cost")

    plt.savefig(f"dbscan_clustering_{scenario}.png")
    plt.close()
    print(f"Plot saved to dbscan_clustering_{scenario}.png")


if __name__ == "__main__":
    execute_clustering(benchmark_name="mfpbench", table_name="data_generation_pd1", only_incumbent=True, use_pca=False)
