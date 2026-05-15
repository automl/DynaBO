import pandas as pd

from dynabo.data_processing.download_all_files import (
    N_REJECTION_SAMPLES_ABLATION_TABLE_PATH,
    N_REJECTION_SAMPLES_ABLATION_INCUMBENT_PATH,
)
from dynabo.plotting.plotting_utils import add_regret, get_min_costs, merge_df

PRIOR_KIND_TITLES = {"good": "Expert", "medium": "Advanced", "misleading": "Misleading", "deceiving": "Deceiving"}
N_SAMPLES_ORDER = [100, 500, 1000]


def load_n_rejection_samples_ablation():
    """
    Load the n_prior_validation_samples ablation data and print a markdown table per
    benchmarklib. Rows are prior kinds (renamed), columns are n_prior_validation_samples
    sorted as 100, 500, 1000.
    """
    table = pd.read_csv(N_REJECTION_SAMPLES_ABLATION_TABLE_PATH)
    incumbents = pd.read_csv(N_REJECTION_SAMPLES_ABLATION_INCUMBENT_PATH)
    config_df, _ = merge_df(table, incumbents, None)

    def fmt(agg: pd.DataFrame) -> pd.Series:
        return agg["mean"].map("{:.4f}".format) + " ± " + agg["sem"].map("{:.4f}".format)

    for benchmarklib in config_df["benchmarklib"].unique():
        bench_df = config_df[config_df["benchmarklib"] == benchmarklib].copy()

        if benchmarklib == "yahpogym":
            bench_df.loc[bench_df["scenario"] == "lcbench", "cost"] /= 100

        min_costs = get_min_costs(benchmarklib=benchmarklib)
        (bench_df,) = add_regret([bench_df], min_costs, benchmarklib=benchmarklib)

        agg = bench_df.groupby(["prior_kind", "n_prior_validation_samples"])["regret"].agg(["mean", "sem"])
        means = agg["mean"].unstack("n_prior_validation_samples")
        summary = fmt(agg).unstack("n_prior_validation_samples")

        # Sort columns and rename index
        available_cols = [c for c in N_SAMPLES_ORDER if c in summary.columns]
        summary = summary[available_cols]
        means = means[available_cols]
        summary.index = summary.index.map(PRIOR_KIND_TITLES)
        means.index = means.index.map(PRIOR_KIND_TITLES)

        # Bold best, italicize second best per row
        for idx in summary.index:
            sorted_cols = means.loc[idx].sort_values().index.tolist()
            best_col, second_col = sorted_cols[0], sorted_cols[1]
            summary.at[idx, best_col] = f"**{summary.at[idx, best_col]}**"
            summary.at[idx, second_col] = f"*{summary.at[idx, second_col]}*"

        print(f"\n## {benchmarklib}\n")
        print(summary.to_markdown())


if __name__ == "__main__":
    load_n_rejection_samples_ablation()
