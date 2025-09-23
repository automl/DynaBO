import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from gower import gower_matrix
from typing import Optional
import numpy as np
import os
from dynabo.utils.data_utils import extract_dataframe_from_column_and_after_n_evaluations, load_df, save_base_table


def execute_clustering_mfpbench(benchmark_name: str, table_name: str):
    scenario_dfs = prepare_data_mfpbench(benchmark_name, table_name, only_incumbent=False)
    for scenario, df in scenario_dfs.items():
        # Use DBSCAN to cluster the data
        labels_agg, df = create_clusters(df, scenario)

        save_clusters(labels_agg, df, scenario, dataset=None)

        # Make first axis bigger
        fig, axes = plt.subplots(1, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [2, 1, 1], "wspace": 0.4})

        create_cluster_plot(df, fig, axes[0], labels_agg, 10, 300)

        size_ax = axes[1]
        create_cluster_size_plot(labels_agg, size_ax)

        cost_ax = axes[2]
        create_cost_plot(df, cost_ax, labels_agg)

        fig.suptitle(f"{scenario}")

        plt.savefig(f"plots/prior_clustering/{benchmark_name}/hierachical_clustering_{scenario}.png", bbox_inches="tight")
        plt.close()

def prepare_data_mfpbench(benchmark_name: str, table_name: str, only_incumbent: bool = False):
    save_base_table(benchmark_name, table_name, only_incumbent=only_incumbent)
    base_df = load_df(benchmark_name="mfpbench")

    scenario_dfs = {}
    for scenario in base_df.scenario.unique():
        local_df = extract_relevant_data(base_df, scenario, dataset=None)
        scenario_dfs[scenario] = local_df

    return scenario_dfs

def execute_clustering_yahpogym(benchmark_name: str, table_name: str):
    scenario_dfs = prepare_data_yahpogym(benchmark_name, table_name, only_incumbent=False)
    for scenario, dataset_dfs in scenario_dfs.items():
        for dataset, df in dataset_dfs.items():
            # Use DBSCAN to cluster the data
            try:
                labels_agg, df = create_clusters(df, scenario)

                save_clusters(labels_agg, df, scenario, dataset, )

                # Make first axis bigger
                fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.4})

                size_ax = axes[0]
                create_cluster_size_plot(labels_agg, size_ax)

                cost_ax = axes[1]
                create_cost_plot(df, cost_ax, labels_agg)

                fig.suptitle(f"{scenario}")

                if not os.path.exists(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}"):
                    os.makedirs(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}")
                plt.savefig(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}/hierarchical_clustering.png", bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Could not cluster {scenario} - {dataset} due to {e}")
                # save empty file to indicate failure
                if not os.path.exists(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}"):
                    os.makedirs(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}")
                with open(f"plots/prior_clustering/{benchmark_name}/{scenario}/{dataset}/hierarchical_clustering_failed.txt", "w") as f:
                    f.write(str(e))
                    f.write("\n")


def prepare_data_yahpogym(benchmark_name: str, table_name: str, only_incumbent: bool = False):
    #save_base_table(benchmark_name, table_name, only_incumbent=only_incumbent, database_name = "dynabo_normal_scale")
    print("Downloaded data")
    base_df = load_df(benchmark_name="yahpogym")
    print("Loaded data")
    scenario_dfs = {}
    for scenario in base_df.scenario.unique():
        dataset_dfs = {}
        relevant_df = base_df[base_df.scenario == scenario]
        for dataset in relevant_df.dataset.unique():
            local_df = extract_relevant_data(relevant_df, scenario, dataset)
            dataset_dfs[dataset] = local_df
        scenario_dfs[scenario] = dataset_dfs

    return scenario_dfs

def extract_relevant_data(base_df: pd.DataFrame, scenario: str, dataset: Optional[str] = None):
        relevant_df = base_df[base_df.scenario == scenario]
        if dataset is not None:
            relevant_df = relevant_df[relevant_df.dataset == dataset]

        incumbents = relevant_df[relevant_df["incumbent"] == True]
        # Sample from the initial design
        non_incumbents = relevant_df[(relevant_df["incumbent"] == False) & (relevant_df["after_n_evaluations"] <= 100)].sample(len(incumbents))
        relevant_df = pd.concat([incumbents, non_incumbents])

        costs = relevant_df[["cost"]].reset_index(drop=True)

        config_df = extract_dataframe_from_column_and_after_n_evaluations(relevant_df[["configuration"]])
        relevant_df = config_df.join(costs, how="left")
        return relevant_df


def create_clusters(df: pd.DataFrame, scenario: str):
    cluster_df = df.copy()
    cluster_df = cluster_df.drop(columns=["cost"])

    cat_features = [not pd.api.types.is_numeric_dtype(cluster_df[col]) for col in cluster_df.columns]

    # cat_mask is a list of booleans: True for categorical-like columns
    distance_matrix = gower_matrix(cluster_df.fillna(-1), cat_features=cat_features)

    agg = AgglomerativeClustering(
        n_clusters=100,
    )
    initial_labels = agg.fit_predict(distance_matrix)

    # Calculate average cost for each cluster
    avg_costs = df.groupby(initial_labels)["cost"].median()

    # Create mapping from old to new labels based on sorted costs
    label_mapping = dict(zip(avg_costs.sort_values().index, range(len(avg_costs))))

    # Apply the mapping to get new labels
    labels_agg = pd.Series(initial_labels).map(label_mapping).values
    df["median_cost"] = df["cost"].groupby(initial_labels).transform("median")
    df["cluster"] = labels_agg

    medoids = {}
    for i in np.unique(labels_agg):
        cluster_idx = np.where(labels_agg == i)[0]
        cluster_distances = distance_matrix[np.ix_(cluster_idx, cluster_idx)]
        medoid = cluster_idx[cluster_distances.sum(axis=1).argmin()]
        medoid_config = df.iloc[medoid].to_dict()
        medoids[i] = medoid_config

    # For each cluster add the medoid configuration to the dataframe
    medoid_df = pd.DataFrame(medoids)
    medoid_df = medoid_df.transpose()
    medoid_df = medoid_df.drop(columns=["cost", "cluster"])
    medoid_df = medoid_df.add_prefix("medoid_")
    medoid_df = medoid_df.reset_index()
    medoid_df = medoid_df.rename(columns={"index": "cluster"})
    df = pd.merge(df, medoid_df, left_on="cluster", right_on="cluster", how="left")

    return labels_agg, df


def save_clusters(labels_agg, df: pd.DataFrame, scenario: str, dataset: Optional[str]):
    df["cluster"] = labels_agg

    # Sort by cluster
    df = df.sort_values(by="cluster")

    # Save to csv
    if dataset is not None:
        df.to_csv(f"benchmark_data/prior_data/yahpogym/cluster/{scenario}/{dataset}.csv", index=False)
    else:
        df.to_csv(f"benchmark_data/prior_data/mfpbench/cluster/{scenario}.csv", index=False)


def create_cluster_plot(df: pd.DataFrame, fig, cluster_ax, labels_agg, min_size, max_size):
    normalized_cost = (df["cost"] - df["cost"].min()) / (df["cost"].max() - df["cost"].min())
    sizes = (1 - normalized_cost) * (max_size - min_size) + min_size

    df = df.drop(columns=["config_lr_power", "config_opt_momentum"])
    reduced_data = df.to_numpy()

    cluster_ax.remove()
    cluster_ax = fig.add_subplot(1, 3, 1, projection="3d")

    # Capture the scatter plot object
    cluster_ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        df["cost"] * -1,  # Using positive cost for a more intuitive Z-axis
        c=labels_agg,
        cmap="viridis",
        s=sizes,
    )

    # Set labels
    cluster_ax.set_xlabel(df.columns[0])
    cluster_ax.set_ylabel(df.columns[1])
    cluster_ax.set_zlabel("Cost")

    # Rotate learning rate axis
    cluster_ax.view_init(elev=10, azim=135)

    # Set titles and labels
    cluster_ax.set_title("Hierachical Clustering")


def create_cluster_size_plot(labels_agg, size_ax):
    cluster_sizes = pd.Series(labels_agg).value_counts()
    unique_labels = sorted(set(labels_agg))
    colors = [plt.cm.viridis(i / len(unique_labels)) for i in unique_labels]

    # Create color mapping
    color_dict = dict(zip(unique_labels, colors))

    # Plot bars in order of cluster number
    for cluster in sorted(cluster_sizes.index):
        size_ax.barh(cluster, cluster_sizes[cluster], color=color_dict[cluster])

    size_ax.set_title("Cluster Size")
    size_ax.set_xlabel("Size")
    size_ax.set_ylabel("Cluster")


def create_cost_plot(df, cost_ax, labels_agg):
    # Prepare data for boxplot
    costs_by_cluster = [df[labels_agg == i]["cost"].values for i in range(len(set(labels_agg)))]
    unique_labels = sorted(set(labels_agg))
    colors = [plt.cm.viridis(i / len(unique_labels)) for i in unique_labels]

    # Create horizontal boxplot
    bp = cost_ax.boxplot(
        costs_by_cluster,
        vert=False,  # Make boxes horizontal
        patch_artist=True,  # Fill boxes with color
        medianprops=dict(color="white"),  # Make median lines white
        flierprops=dict(marker="o", markerfacecolor="white", markersize=4),  # Outlier style
        positions=range(len(unique_labels)),
    )  # Position boxes at cluster numbers

    # Color the boxes according to the cluster colors
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    cost_ax.set_title("Cost Distribution per Cluster")
    cost_ax.set_xlabel("Cost")
    cost_ax.set_ylabel("Cluster")

    # Set y-axis limits to show all clusters
    cost_ax.set_ylim(-1, len(unique_labels))


if __name__ == "__main__":
    #execute_clustering_mfpbench(benchmark_name="mfpbench", table_name="data_generation_pd1")
    execute_clustering_yahpogym(benchmark_name="yahpogym", table_name="data_generation_yahpo")
