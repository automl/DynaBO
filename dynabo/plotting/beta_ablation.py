import numpy as np
import pandas as pd

from dynabo.data_processing.download_all_files import (
    BETA_ABLATION_INCUMBENT_PATH,
    BETA_ABLATION_PRIOR_PATH,
    BETA_ABLATION_TABLE_PATH,
)
from dynabo.plotting.plotting_utils import (
    add_regret,
    create_all_scenarios_plot,
    create_overall_plot,
    create_scenario_plots,
    filter_prior_approach,
    get_min_costs,
    merge_df,
)

BASE_PATH = "plots/beta_ablation"

# Beta is stored as prior_decay_enumerator / prior_decay_denominator, the denominator is 1 in this ablation.
BETAS = [1, 2, 5, 10, 20]

PRIOR_KIND_TITLES = {"good": "Expert", "medium": "Advanced", "misleading": "Misleading", "deceiving": "Deceiving"}

# Okabe-Ito colors, ordered from small to large beta. #E69F00 is left out, it marks the prior positions.
STYLE_DICT = {
    r"$\beta$ = 1": {"color": "#000000", "marker": "o", "linestyle": (0, ())},
    r"$\beta$ = 2": {"color": "#0072B2", "marker": "s", "linestyle": (0, (5, 1))},
    r"$\beta$ = 5": {"color": "#009E73", "marker": "d", "linestyle": (0, (3, 5, 1, 5))},
    r"$\beta$ = 10": {"color": "#D55E00", "marker": "v", "linestyle": (0, (1, 1))},
    r"$\beta$ = 20": {"color": "#CC79A7", "marker": "^", "linestyle": (0, (3, 1, 1, 1, 1, 1))},
}


def load_beta_ablation():
    """
    Load the cost data of the beta ablation, saved in the filesystem, and add regret.
    """
    prior_table = pd.read_csv(BETA_ABLATION_TABLE_PATH)
    prior_configs = pd.read_csv(BETA_ABLATION_INCUMBENT_PATH)
    prior_priors = pd.read_csv(BETA_ABLATION_PRIOR_PATH)
    prior_config_df, prior_priors_df = merge_df(prior_table, prior_configs, prior_priors)

    min_costs = get_min_costs(benchmarklib="mfpbench")
    prior_config_df, prior_priors_df = add_regret([prior_config_df, prior_priors_df], min_costs, benchmarklib="mfpbench")

    return prior_config_df, prior_priors_df


def split_by_beta(prior_config_df: pd.DataFrame, prior_prior_df: pd.DataFrame, validate_prior: bool = True):
    """
    Split the runs into one incumbent dataframe per beta, keeping all other approach parameters fixed.
    """
    config_dict = {}
    for beta in BETAS:
        incumbent_df, _ = filter_prior_approach(
            incumbent_df=prior_config_df,
            prior_df=prior_prior_df,
            select_dynabo=True,
            select_pibo=False,
            prior_decay_enumerator=beta,
            prior_std_denominator=5,
            prior_static_position=True,
            prior_every_n_trials=10,
            validate_prior=validate_prior,
            n_prior_based_samples=0,
            prior_validation_method="difference" if validate_prior else None,
            prior_validation_manwhitney_p=None,
            prior_validation_difference_threshold=-0.15 if validate_prior else None,
            remove_old_priors=False,
            prior_decay="linear",
        )
        config_dict[rf"$\beta$ = {beta}"] = incumbent_df

    return config_dict


def plot_beta_ablation():
    prior_config_df, prior_prior_df = load_beta_ablation()
    config_dict = split_by_beta(prior_config_df, prior_prior_df)

    # The priors sit at the same positions for every beta, so a single entry is enough to mark them. The key
    # has to be `DynaBO`, that is what `plot_final_run` looks for.
    prior_dict = {"DynaBO": prior_prior_df}

    scenarios = sorted(prior_config_df["scenario"].unique())

    for y_column in ["regret", "cost"]:
        create_scenario_plots(
            config_dict,
            prior_dict,
            STYLE_DICT,
            error_bar_type="se",
            scenarios=scenarios,
            benchmarklib="mfpbench",
            base_path=BASE_PATH,
            ncol=len(STYLE_DICT),
            y_column=y_column,
        )
        create_overall_plot(
            config_dict,
            prior_dict,
            STYLE_DICT,
            error_bar_type="se",
            benchmarklib="mfpbench",
            base_path=BASE_PATH,
            ncol=len(STYLE_DICT),
            y_column=y_column,
        )
        create_all_scenarios_plot(
            config_dict,
            prior_dict,
            STYLE_DICT,
            error_bar_type="se",
            scenarios=scenarios,
            benchmarklib="mfpbench",
            base_path=BASE_PATH,
            ncol=len(STYLE_DICT),
            y_column=y_column,
        )


def collect_runs(validate_prior: bool) -> pd.DataFrame:
    """
    One row per finished run of the given validation arm, with the beta it belongs to.

    `final_regret` is a keyfield level result that is repeated across the incumbents of a run, so the
    incumbent rows are collapsed to one row per run.
    """
    prior_config_df, prior_prior_df = load_beta_ablation()
    config_dict = split_by_beta(prior_config_df, prior_prior_df, validate_prior=validate_prior)

    runs = [df.drop_duplicates(subset=["experiment_id"]).assign(beta=beta) for beta, df in zip(BETAS, config_dict.values())]
    runs = pd.concat(runs, ignore_index=True)
    runs["validate_prior"] = validate_prior
    return runs[["beta", "validate_prior", "scenario", "seed", "prior_kind", "final_regret"]]


def _format(values: pd.Series) -> str:
    if len(values) == 0:
        return "-"
    return f"{values.mean():.4f} ± {values.std(ddof=1) / np.sqrt(len(values)):.4f}"


PRIOR_KIND_COLUMNS = list(PRIOR_KIND_TITLES.items()) + [(None, "All")]


def _regret_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Mean final regret +- standard error, one row per beta and one column per prior kind.
    """
    rows = []
    for beta in BETAS:
        beta_runs = runs[runs["beta"] == beta]
        row = {"beta": beta}
        row.update({title: _format(beta_runs["final_regret"] if prior_kind is None else beta_runs[beta_runs["prior_kind"] == prior_kind]["final_regret"]) for prior_kind, title in PRIOR_KIND_COLUMNS})
        rows.append(row)

    return pd.DataFrame(rows)


def print_beta_ablation_table():
    """
    Print the final regret per beta and prior kind, once for runs with and once without prior rejection.
    """
    for validate_prior in [True, False]:
        runs = collect_runs(validate_prior=validate_prior)
        print(f"\n## {'With' if validate_prior else 'Without'} prior rejection ({len(runs)} runs)\n")
        print(_regret_table(runs).to_markdown(index=False))


if __name__ == "__main__":
    plot_beta_ablation()
    print_beta_ablation_table()
