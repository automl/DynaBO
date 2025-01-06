import ast
import os

import pandas as pd
from py_experimenter.experimenter import PyExperimenter


def create_prior_data_path(scenario: str, dataset: str, metric: str):
    if not os.path.exists(os.path.join("benchmark_data", "prior_data", scenario, dataset, metric)):
        os.makedirs(os.path.join("benchmark_data", "prior_data", scenario, dataset, metric))
    return os.path.join("benchmark_data", "prior_data", scenario, dataset, metric)


def connect_to_database() -> PyExperimenter:
    EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
    DB_CRED_FILE_PATH = "config/database_credentials.yml"

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )

    return experimenter


def save_base_table():
    experimenter = connect_to_database()
    table = get_table(experimenter)
    return table


def save_table(table: pd.DataFrame, path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    table.to_csv(path)


def get_table(experimenter: PyExperimenter):
    return experimenter.get_table()


def load_df(path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    return pd.read_csv(path)


def get_experiment(df: pd.DataFrame, scenario: str, dataset: str, metric: str):
    return df[(df.scenario == scenario) & (df.dataset == dataset) & (df.metric == metric)]


def build_prior_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, that is allready reduced to just one scenarion, dataset, metric combiantion, this function
    extracts the columns encoded in `incumbent_trace` into seperate columns.
    """
    seed_dfs = {}
    for seed in df["seed"].unique():
        seed_df = df[df.seed == seed]
        extracted_df = extract_dataframe_from_column(seed_df, "incumbent_trace")

        # Create Multiple lines in the seed_df for each entry in the extracted_df
        seed_df = seed_df.drop(columns=["incumbent_trace"])
        assert len(seed_df) == 1
        seed_df = pd.concat([seed_df] * len(extracted_df), ignore_index=True)

        # Add the extracted columns to the seed_df
        for key in extracted_df.columns:
            seed_df[key] = extracted_df[key].values

        seed_dfs[seed] = seed_df

    return pd.concat(seed_dfs.values(), ignore_index=True)


def extract_dataframe_from_column(df, column_name):
    """
    Extracts a structured DataFrame from a column containing string representations of trace data.

    Parameters:
    - df: The original pandas DataFrame.
    - column_name: Name of the column containing string trace data.

    Returns:
    - A new DataFrame with each attribute of the trace data in its own column.
    """
    # Replace nan with "None" to avoid errors
    trace_data = df[column_name].apply(ast.literal_eval)

    # Flatten the data and create a new DataFrame
    rows = []
    for trace_list in trace_data:
        for entry in trace_list:
            time_found, accuracy, no_evaluation, config = entry
            row = {
                "time_found": time_found,
                "accuracy": accuracy,
                "no_evaluation": no_evaluation,
            }
            for key, value in config.items():
                row[f"config_{key}"] = value

            rows.append(row)

    return pd.DataFrame(rows)
