import ast

import pandas as pd
from py_experimenter.experimenter import PyExperimenter


def connect_to_database() -> PyExperimenter:
    EXP_CONFIG_FILE_PATH = "config/experiment_config.yml"
    DB_CRED_FILE_PATH = "config/database_credentials.yml"

    experimenter = PyExperimenter(
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,
        database_credential_file_path=DB_CRED_FILE_PATH,
        use_codecarbon=False,
    )

    return experimenter


def save_table(table: pd.DataFrame, path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    table.to_csv(path)


def get_table(experimenter: PyExperimenter):
    return experimenter.get_table()


def load_table(path: str = "benchmark_data/gt_prior_data/origin_table.csv"):
    return pd.read_csv(path)


def get_single_run(df: pd.DataFrame, scenario: str, dataset: str, metric: str, seed: int):
    experiment = get_experiment(df, scenario, dataset, metric)
    return experiment[experiment.seed == seed]


def get_experiment(df: pd.DataFrame, sceanrio: str, dataset: str, metric: str):
    return df[(df.scenario == sceanrio) & (df.dataset == dataset) & (df.metric == metric)]


def extract_dataframe_from_column(df, column_name):
    """
    Extracts a structured DataFrame from a column containing string representations of trace data.

    Parameters:
    - df: The original pandas DataFrame.
    - column_name: Name of the column containing string trace data.

    Returns:
    - A new DataFrame with each attribute of the trace data in its own column.
    """
    # Parse the string column into Python objects
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
            row.update(config)
            rows.append(row)

    return pd.DataFrame(rows)
