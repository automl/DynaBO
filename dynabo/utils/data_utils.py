import ast
import os

import pandas as pd
from py_experimenter.experimenter import PyExperimenter


def create_prior_data_path(scenario: str, dataset: str, metric: str):
    if not os.path.exists(os.path.join("benchmark_data", "prior_data", scenario, str(dataset), metric)):
        os.makedirs(os.path.join("benchmark_data", "prior_data", scenario, str(dataset), metric))
    return os.path.join("benchmark_data", "prior_data", scenario, str(dataset), metric)


def connect_to_database() -> PyExperimenter:
    EXP_CONFIG_FILE_PATH = "dynabo/experiments/gt_experiments/config.yml"
    DB_CRED_FILE_PATH = "config/database_credentials.yml"

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )

    return experimenter


def save_base_table():
    experimenter = connect_to_database()
    base_table = experimenter.get_table()
    configs = experimenter.get_logtable("configs")
    configs.drop(columns=["ID"])
    table = base_table.merge(configs, how="left", left_on=["ID"], right_on=["experiment_id"])
    save_table(table)
    return table


def save_table(table: pd.DataFrame, path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    table.to_csv(path, index=False)


def load_df(path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    return pd.read_csv(path)


def get_experiment(df: pd.DataFrame, scenario: str, dataset: str, metric: str):
    return df[(df.scenario == scenario) & (df.dataset == dataset) & (df.metric == metric)]


def build_prior_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, that is allready reduced to just one scenarion, dataset, metric combiantion, this function
    extracts the columns encoded in `incumbent_trace` into seperate columns.
    """
    df = df.reset_index(drop=True)
    extracted_df = extract_dataframe_from_column(df)
    df = df.drop(columns=["configuration"])
    if len(df) != len(extracted_df):
        print()
    expanded_df = pd.concat([df, extracted_df], axis=1)
    if len(expanded_df) != len(extracted_df):
        print
    return expanded_df


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
