import json
from copy import deepcopy

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from matplotlib import pyplot as plt
from py_experimenter.experimenter import PyExperimenter
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS

from dynabo.utils.data_utils import extract_dataframe_from_column_and_after_n_evaluations


def build_experimenter() -> PyExperimenter:
    return PyExperimenter(
        experiment_configuration_file_path="dynabo/experiments/baseline_experiments/config.yml",
        table_name="one_run",
    )


def extract_relevant_data(base_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    scenario_df = base_df[base_df.scenario == scenario]
    costs = scenario_df[["cost"]].reset_index(drop=True)
    config_df = extract_dataframe_from_column_and_after_n_evaluations(scenario_df[["configuration"]])
    scenario_df = config_df.join(costs, how="left")
    return scenario_df


def load_configspace() -> ConfigurationSpace:
    configspace_json = """{
    "name": null,
    "hyperparameters": [
        {
            "type": "uniform_float",
            "name": "lr_decay_factor",
            "lower": 0.010294,
            "upper": 0.989753,
            "default_value": 0.5000235,
            "log": false,
            "meta": null
        },
        {
            "type": "uniform_float",
            "name": "lr_initial",
            "lower": 1e-05,
            "upper": 9.774312,
            "default_value": 0.009886512024,
            "log": true,
            "meta": null
        },
        {
            "type": "uniform_float",
            "name": "lr_power",
            "lower": 0.100225,
            "upper": 1.999326,
            "default_value": 1.0497755,
            "log": false,
            "meta": null
        },
        {
            "type": "uniform_float",
            "name": "opt_momentum",
            "lower": 5.9e-05,
            "upper": 0.998993,
            "default_value": 0.0076772773169,
            "log": true,
            "meta": null
        }
    ],
    "conditions": [],
    "forbiddens": [],
    "python_module_version": "1.2.0",
    "format_version": 0.4
}"""
    configspace = json.loads(configspace_json)
    return ConfigurationSpace.from_serialized_dict(configspace)


def generate_edge_datapoints(configspace: ConfigurationSpace):
    edge_datapoints = list()
    data_bound_names = list()
    for number in range(2 ** len(configspace.values())):
        datapoint = list()
        data_bounds = list()
        for i, hyperparameter in enumerate(configspace.values()):
            if number & (1 << i):
                datapoint.append(hyperparameter.upper)
                data_bounds.append("upper")
            else:
                datapoint.append(hyperparameter.lower)
                data_bounds.append("lower")
        edge_datapoints.append(datapoint)
        data_bound_names.append(data_bounds)
    edge_datapoints
    data_bound_names
    return edge_datapoints, data_bound_names


def sample_random_configs(configspace: ConfigurationSpace, num_samples: int):
    random_sampled_configs = configspace.sample_configuration(num_samples)
    random_sampled_configs = [list(x.values()) for x in random_sampled_configs]
    random_sampled_configs
    return random_sampled_configs


def train_random_forest(relevant_data: pd.DataFrame) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    X = np.array(relevant_data.drop(columns=["cost"]))
    y = np.array(relevant_data["cost"])
    rf.fit(X, y)
    return rf


def build_configs_matrix(relevant_data: pd.DataFrame, edge_datapoints, random_sampled_configs):
    optimal_config = [0.2303439554455, 0.3872140985972, 0.5081353075926, 0.9154046565785]
    configs = deepcopy(relevant_data)
    configs = configs.drop(columns=["cost"])
    configs = [list(x) for x in configs.to_numpy()]
    n_found_configs = len(configs)
    n_edge_datapoints = len(edge_datapoints)
    n_random_sampled_configs = len(random_sampled_configs)
    configs += edge_datapoints
    configs += random_sampled_configs
    configs.append(optimal_config)
    return configs, n_found_configs, n_edge_datapoints, n_random_sampled_configs


def reduce_dimensions(configs):
    mds = MDS(n_components=2)
    reduced_dimension = mds.fit_transform(configs)
    return reduced_dimension


def plot_runs_and_points(original_configs, reduced_dimension, table, n_edge_datapoints, n_random_sampled_configs, rf):
    for run in range(len(table)):
        plt.figure(figsize=(10, 10))
        plt.title(f"Run {run}")
        
        found_original_configs = original_configs[50 * run : 50 * (run + 1)]
        found_original_configs_predictions = rf.predict(found_original_configs)
        edge_original_configs = original_configs[500 : 500 + n_edge_datapoints]
        edge_original_configs_predictions = rf.predict(edge_original_configs)
        random_original_configs = original_configs[500 + n_edge_datapoints : 500 + n_edge_datapoints + n_random_sampled_configs]
        random_original_configs_predictions = rf.predict(random_original_configs)

        found_configs_reduced = reduced_dimension[50 * run : 50 * (run + 1)]
        edge_points_reduced = reduced_dimension[500 : 500 + n_edge_datapoints]
        random_points_reduced = reduced_dimension[500 + n_edge_datapoints : 500 + n_edge_datapoints + n_random_sampled_configs]
        optimal_point_reduced = reduced_dimension[-1:]

        pcolor_configs = np.concatenate((found_configs_reduced, edge_points_reduced, random_points_reduced))
        pcolor_configs_predictions = np.concatenate((found_original_configs_predictions, edge_original_configs_predictions, random_original_configs_predictions))

        tcf = plt.tricontourf(
            pcolor_configs[:, 0],
            pcolor_configs[:, 1],
            pcolor_configs_predictions,
            levels=100,
            cmap="viridis",
        )
        plt.colorbar(tcf)

        plt.scatter(found_configs_reduced[:, 0], found_configs_reduced[:, 1], c="green", marker="o")
        plt.scatter(edge_points_reduced[:, 0], edge_points_reduced[:, 1], c="red", marker="o")
        plt.scatter(random_points_reduced[:, 0], random_points_reduced[:, 1], c="blue", marker="o")
        plt.scatter(optimal_point_reduced[:, 0], optimal_point_reduced[:, 1], c="black", marker="x", s=100)
        plt.savefig(f"run_{run}.png")


def plot_all_points(reduced_dimension, relevant_data, edge_datapoints, random_sampled_configs):
    plt.figure(figsize=(10, 10))
    colors = ["green"] * len(relevant_data)
    for i in range(len(edge_datapoints)):
        colors.append("red")
    for i in range(len(random_sampled_configs)):
        colors.append("blue")
    colors.append("black")
    plt.scatter(reduced_dimension[:, 0], reduced_dimension[:, 1], c=colors)
    plt.colorbar()
    plt.savefig("all_points.png")


def main():
    experimenter = build_experimenter()
    table = experimenter.get_table()
    log_table = experimenter.get_logtable("configs")
    log_table

    df = pd.merge(table, log_table, left_on="ID", right_on="experiment_id")
    relevant_data = extract_relevant_data(df, "cifar100_wideresnet_2048")
    relevant_data

    configspace = load_configspace()
    edge_datapoints, data_bound_names = generate_edge_datapoints(configspace)
    random_sampled_configs = sample_random_configs(configspace, 50)

    rf = train_random_forest(relevant_data)

    original_configs, n_found_configs, n_edge_datapoints, n_random_sampled_configs = build_configs_matrix(
        relevant_data, edge_datapoints, random_sampled_configs
    )
    reduced_dimension = reduce_dimensions(original_configs)

    plot_runs_and_points(original_configs, reduced_dimension, table, n_edge_datapoints, n_random_sampled_configs, rf)


if __name__ == "__main__":
    main()
