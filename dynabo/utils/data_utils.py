import ast
import os
from copy import deepcopy
from typing import Optional

import pandas as pd
from py_experimenter.experimenter import PyExperimenter


def create_prior_data_path_yahpo(scenario: str, dataset: str, metric: str):
    if not os.path.exists(os.path.join("benchmark_data", "prior_data", "yahpogym", scenario, str(dataset), metric)):
        os.makedirs(os.path.join("benchmark_data", "prior_data", "yahpogym", scenario, str(dataset), metric))
    return os.path.join("benchmark_data", "prior_data", "yahpogym", scenario, str(dataset), metric)


def create_prior_data_path_pd1(scenario: str):
    if not os.path.exists(os.path.join("benchmark_data", "prior_data", "mfpbench", scenario)):
        os.makedirs(os.path.join("benchmark_data", "prior_data", "mfpbench", scenario))
    return os.path.join("benchmark_data", "prior_data", "mfpbench", scenario)


def connect_to_database(table_name: str) -> PyExperimenter:
    EXP_CONFIG_FILE_PATH = "dynabo/experiments/baseline_experiments/config.yml"
    DB_CRED_FILE_PATH = "config/database_credentials.yml"

    experimenter = PyExperimenter(experiment_configuration_file_path=EXP_CONFIG_FILE_PATH, database_credential_file_path=DB_CRED_FILE_PATH, use_codecarbon=False, table_name=table_name)

    return experimenter


def save_base_table(benchmark_name: str, table_name: str, only_incumbent: bool = False):
    experimenter = connect_to_database(table_name)
    base_table = experimenter.get_table()
    if only_incumbent:
        configs = experimenter.get_logtable("configs", condition="incumbent = 1")
    else:
        configs = experimenter.get_logtable("configs")
    configs = configs.drop(columns=["ID"])
    table = base_table.merge(configs, how="left", left_on=["ID"], right_on=["experiment_id"])
    path = f"benchmark_data/gt_prior_data/{benchmark_name}/origin_table.csv"
    save_table(table, path)
    return table


def save_table(table: pd.DataFrame, path: str):
    table.to_csv(path, index=False)


def load_df(benchmark_name: str = "yahpogym") -> pd.DataFrame:
    path = f"benchmark_data/gt_prior_data/{benchmark_name}/origin_table.csv"
    return pd.read_csv(path)


def get_experiment(df: pd.DataFrame, scenario: str, dataset: str, metric: str):
    return df[(df.scenario == scenario) & (df.dataset == dataset) & (df.metric == metric)]


def build_prior_dataframe(df: pd.DataFrame, filter_with_epsilon_distance: bool, filter_epsilon: Optional[float]) -> pd.DataFrame:
    """
    Given a dataframe, that is allready reduced to just one scenarion, dataset, metric combiantion, this function
    extracts the columns encoded in `incumbent_trace` into seperate columns.

    If `filter_with_epsilon_distance` is set to True, the dataframe is filtered to only
    """
    df = df.reset_index(drop=True)
    df = df[df["incumbent"] == 1]
    if filter_with_epsilon_distance:
        df = filter_incumbents(df, filter_epsilon)
    extracted_df = extract_dataframe_from_column(df)
    df = df.drop(columns=["configuration"])
    expanded_df = pd.concat([df, extracted_df], axis=1)
    expanded_df = expanded_df.drop(columns=["error"])
    return expanded_df


def filter_incumbents(df: pd.DataFrame, filter_epsilon: float) -> pd.DataFrame:
    """
    Filters the dataframe to only include rows where the performance is outside a certain epsilon distance from all other rows.
    """
    if df["scenario"].nunique() > 1:
        raise ValueError("The dataframe contains multiple scenarios. Please filter the dataframe to a single scenario before applying this function.")
    elif df["scenario"][0] == "lcbench":
        filter_epsilon *= 100
    df = df.sort_values("cost", ascending=False)
    filtered_df = deepcopy(df)
    max_cost = df["cost"].max()
    min_cost = df["cost"].min()
    for index, row in df.iterrows():
        if row["cost"] > (max_cost - filter_epsilon) or row["cost"] < (min_cost + filter_epsilon):
            filtered_df = filtered_df.drop(index)
        else:
            max_cost = row["cost"]
    return filtered_df.reset_index(drop=True)


def extract_dataframe_from_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a structured DataFrame from a column containing string representations of trace data.

    Parameters:
    - df: The original pandas DataFrame.
    - column_name: Name of the column containing string trace data.

    Returns:
    - A new DataFrame with each attribute of the trace data in its own column.
    """
    df["configuration"] = df["configuration"].apply(ast.literal_eval)  # safer than eval
    dict_df = pd.json_normalize(df["configuration"])
    dict_df.columns = [f"config_{column}" for column in dict_df.columns]
    dict_df.reset_index(drop=True)

    return dict_df


def extract_dataframe_from_column_and_after_n_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    configuration_df = df["configuration"].apply(ast.literal_eval)  # safer than eval
    configuration_df = pd.json_normalize(configuration_df)
    configuration_df.columns = [f"config_{column}" for column in configuration_df.columns]
    return configuration_df
