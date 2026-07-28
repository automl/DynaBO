from typing import Optional

import pandas as pd
from py_experimenter.experimenter import PyExperimenter

from dynabo.utils.evaluator import yahpo_objective_normalizer

DATA_GENERATION_CONFIG_PATH = "dynabo/experiments/gt_experiments/config.yml"
BASELINE_CONFIG_PATH = "dynabo/experiments/baseline_experiments/config.yml"
PRIOR_EXPERIMENTS_PATH = "dynabo/experiments/prior_experiments/config.yml"
CREDENTIALS_PATH = "config/database_credentials.yml"

YHAPO_DATA_GENERATION_ONE_SEED_PATH = "plotting_data/yahpo/datageneration.csv"
YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH = "plotting_data/yahpo/datageneration_incumbent.csv"

YAHPO_DATA_GENERATION_MEDIUM_HARD_PATH = "plotting_data/yahpo/datageneration_medium_hard.csv"
YAHPO_DATA_GENERATION_INCUMBENT_MEDIUM_HARD_PATH = "plotting_data/yahpo/datageneration_incumbent_medium_hard.csv"

YAHPO_BASELINE_TABLE_PATH = "plotting_data/yahpogym/baseline.csv"
YAHPO_BASELINE_INCUMBENT_PATH = "plotting_data/yahpogym/baseline_incumbent.csv"
YAHPO_PRIOR_TABLE_PATH = "plotting_data/yahpogym/prior.csv"
YAHPO_PRIOR_INCUMBENT_PATH = "plotting_data/yahpogym/prior_incumbent.csv"
YAHPO_PRIOR_PRIORS_PATH = "plotting_data/yahpogym/prior_priors.csv"

YAHPO_ABLATION_TABLE_PATH = "plotting_data/yahpogym/yahpo_ablation.csv"
YAHPO_ABLATION_INCUMBENT_PATH = "plotting_data/yahpogym/yahpo_ablation_incumbent.csv"
YAHPO_ABLATION_PRIOR_PATH = "plotting_data/yahpogym/yahpo_ablation_priors.csv"

YAHPO_PRIOR_DECAY_ABLATION_TABLE_PATH = "plotting_data/yahpogym/decay_ablation.csv"
YAHPO_PRIOR_DECAY_ABLATION_INCUMBENT_PATH = "plotting_data/yahpogym/decay_ablation_incumbent.csv"
YAHPO_PRIOR_DECAY_ABLATION_PRIOR_PATH = "plotting_data/yahpogym/decay_ablation_priors.csv"

N_REJECTION_SAMPLES_ABLATION_TABLE_PATH = "plotting_data/yahpogym/n_rejection_samples_ablation.csv"
N_REJECTION_SAMPLES_ABLATION_INCUMBENT_PATH = "plotting_data/yahpogym/n_rejection_samples_ablation_incumbent.csv"
N_REJECTION_SAMPLES_ABLATION_PRIOR_PATH = "plotting_data/yahpogym/n_rejection_samples_ablation_priors.csv"

PRIOR_COMBINATION_ABLATION_TABLE_PATH = "plotting_data/pd1/prior_combination_ablation.csv"
PRIOR_COMBINATION_ABLATION_INCUMBENT_PATH = "plotting_data/pd1/prior_combination_ablation_incumbent.csv"
PRIOR_COMBINATION_ABLATION_PRIOR_PATH = "plotting_data/pd1/prior_combination_ablation_priors.csv"

PRIOR_DECAY_ABLATION_TABLE_PATH = "plotting_data/pd1/decay_ablation.csv"
PRIOR_DECAY_ABLATION_INCUMBENT_PATH = "plotting_data/pd1/decay_ablation_incumbent.csv"
PRIOR_DECAY_ABLATION_PRIOR_PATH = "plotting_data/pd1/decay_ablation_priors.csv"

REMOVE_OLD_PRIORS_ABLATION_TABLE_PATH = "plotting_data/pd1/remove_old_priors_ablation.csv"
REMOVE_OLD_PRIORS_ABLATION_INCUMBENT_PATH = "plotting_data/pd1/remove_old_priors_ablation_incumbent.csv"
REMOVE_OLD_PRIORS_ABLATION_PRIOR_PATH = "plotting_data/pd1/remove_old_priors_ablation_priors.csv"

MIXED_PRIORS_TABLE_PATH = "plotting_data/pd1/mixed_priors.csv"
MIXED_PRIORS_INCUMBENT_PATH = "plotting_data/pd1/mixed_priors_incumbent.csv"
MIXED_PRIORS_PRIORS_PATH = "plotting_data/pd1/mixed_priors_priors.csv"

# Random Forest DAta
RF_PD1_BASELINE_TABLE_PATH = "plotting_data/pd1/rf/baseline.csv"
RF_PD1_BASELINE_INCUMBENT_PATH = "plotting_data/pd1/rf/baseline_incumbent.csv"
RF_PD1_PRIOR_TABLE_PATH = "plotting_data/pd1/rf/prior.csv"
RF_PD1_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/rf/prior_incumbent.csv"
RF_PD1_PRIOR_PRIORS_PATH = "plotting_data/pd1/rf/prior_priors.csv"
RF_PD1_DYNAMIC_PRIORS_TABLE_PATH = "plotting_data/pd1/rf/dynamic_priors.csv"
RF_PD1_DYNAMIC_PRIORS_INCUMBENT_PATH = "plotting_data/pd1/rf/dynamic_priors_incumbent.csv"
RF_PD1_DYNAMIC_PRIORS_PRIORS_PATH = "plotting_data/pd1/rf/dynamic_priors_priors.csv"

# Run PD1 Misleading for Longer
RF_PD1_BASELINE_DECEIVING_LONGER_PATH = "plotting_data/pd1/rf/deceiving_longer/baseline.csv"
RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH = "plotting_data/pd1/rf/deceiving_longer/baseline_incumbent.csv"
RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH = "plotting_data/pd1/rf/deceiving_longer/prior.csv"
RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/rf/deceiving_longer/prior_incumbent.csv"
RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH = "plotting_data/pd1/rf/deceiving_longer/prior_priors.csv"

GP_PD1_BASELINE_DECEIVING_LONGER_PATH = "plotting_data/pd1/gp/deceiving_longer/baseline.csv"
GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH = "plotting_data/pd1/gp/deceiving_longer/baseline_incumbent.csv"
GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH = "plotting_data/pd1/gp/deceiving_longer/prior.csv"
GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/gp/deceiving_longer/prior_incumbent.csv"
GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH = "plotting_data/pd1/gp/deceiving_longer/prior_priors.csv"

# Gaussian Process Data
GP_PD1_BASELINE_TABLE_PATH = "plotting_data/pd1/gp/baseline.csv"
GP_PD1_BASELINE_INCUMBENT_PATH = "plotting_data/pd1/gp/baseline_incumbent.csv"
GP_PD1_PRIOR_TABLE_PATH = "plotting_data/pd1/gp/prior.csv"
GP_PD1_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/gp/prior_incumbent.csv"
GP_PD1_PRIOR_PRIORS_PATH = "plotting_data/pd1/gp/prior_priors.csv"

# LCB Data
LCB_PD1_BASELINE_TABLE_PATH = "plotting_data/pd1/lcb/baseline.csv"
LCB_PD1_BASELINE_INCUMBENT_PATH = "plotting_data/pd1/lcb/baseline_incumbent.csv"
LCB_PD1_PRIOR_TABLE_PATH = "plotting_data/pd1/lcb/prior.csv"
LCB_PD1_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/lcb/prior_incumbent.csv"
LCB_PD1_PRIOR_PRIORS_PATH = "plotting_data/pd1/lcb/prior_priors.csv"

# LCB Yahpo Data
LCB_YAHPO_BASELINE_TABLE_PATH = "plotting_data/yahpogym/lcb/baseline.csv"
LCB_YAHPO_BASELINE_INCUMBENT_PATH = "plotting_data/yahpogym/lcb/baseline_incumbent.csv"
LCB_YAHPO_PRIOR_TABLE_PATH = "plotting_data/yahpogym/lcb/prior.csv"
LCB_YAHPO_PRIOR_INCUMBENT_PATH = "plotting_data/yahpogym/lcb/prior_incumbent.csv"
LCB_YAHPO_PRIOR_PRIORS_PATH = "plotting_data/yahpogym/lcb/prior_priors.csv"


def normalize_legacy_yahpo_costs(table: pd.DataFrame, *logtables: pd.DataFrame):
    """
    Put costs of runs recorded before the evaluator normalization onto the 0..1 objective scale.

    Runs executed before `YAHPOGymEvaluator` divided `val_accuracy` by 100 stored lcbench costs
    on the 0..100 percentage scale, while `acc`-based scenarios were already on 0..1. The scale
    is decided per run via the `metric` keyfield, so the rescale is applied to the matching rows
    of the main table (`final_cost`) and, via `experiment_id`, of the logtables (`cost`).

    Only pass tables from *legacy* runs -- runs executed after the change already log 0..1 and
    would be scaled down a second time.
    """
    table = table.copy()
    logtables = [logtable.copy() for logtable in logtables]

    for metric in table["metric"].unique():
        normalizer = yahpo_objective_normalizer(metric)
        if normalizer == 1.0:
            continue

        rows = table["metric"] == metric
        table.loc[rows, "final_cost"] /= normalizer

        experiment_ids = set(table.loc[rows, "ID"])
        for logtable in logtables:
            log_rows = logtable["experiment_id"].isin(experiment_ids)
            logtable.loc[log_rows, "cost"] /= normalizer

    return (table, *logtables)


def download_prior_tables(database_name: str, table_name: str, approach: Optional[str] = None):
    """
    Download the main table and the `configs`/`priors` logtables of a prior experiment table.

    If `approach` is given (`"dynabo"` or `"pibo"`), only the runs of that approach and their
    log entries are kept.
    """
    experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name=database_name, table_name=table_name)

    table = experimenter.get_table()
    incumbents = experimenter.get_logtable("configs", condition="incumbent = 1")
    priors = experimenter.get_logtable("priors")

    if approach is not None:
        table = table[table[approach] == True]
        experiment_ids = set(table["ID"])
        incumbents = incumbents[incumbents["experiment_id"].isin(experiment_ids)]
        priors = priors[priors["experiment_id"].isin(experiment_ids)]

    return table, incumbents, priors


def concat_prior_tables(*sources):
    """
    Concatenate prior tables coming from different databases.

    The experiment ids are only unique within a single database, so they are offset per source.
    Otherwise `merge_df` would join the incumbents/priors of one source onto the runs of another.
    """
    tables, incumbent_dfs, prior_dfs = [], [], []
    offset = 0
    for table, incumbents, priors in sources:
        table, incumbents, priors = table.copy(), incumbents.copy(), priors.copy()
        table["ID"] += offset
        incumbents["experiment_id"] += offset
        priors["experiment_id"] += offset
        offset = int(table["ID"].max()) + 1

        tables.append(table)
        incumbent_dfs.append(incumbents)
        prior_dfs.append(priors)

    return pd.concat(tables, ignore_index=True), pd.concat(incumbent_dfs, ignore_index=True), pd.concat(prior_dfs, ignore_index=True)


def download_yahpo_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="dynabo_normal_scale", table_name="baseline_yahpo")

    # The baseline and piBO runs predate the objective normalization in `YAHPOGymEvaluator`, so
    # their lcbench costs are still on the 0..100 scale and are rescaled here. The DynaBO runs in
    # `dynabo_sum` are left alone: its lcbench runs are being redone with the normalizing
    # evaluator, and its `acc`-based scenarios were never on the percentage scale to begin with.
    baseline_table, baseline_incumbents = normalize_legacy_yahpo_costs(baseline_experimenter.get_table(), baseline_experimenter.get_logtable("configs"))
    baseline_table.to_csv(YAHPO_BASELINE_TABLE_PATH, index=False)
    baseline_incumbents.to_csv(YAHPO_BASELINE_INCUMBENT_PATH, index=False)

    # The DynaBO runs were repeated in `dynabo_sum`, but that table only contains DynaBO runs.
    # The piBO baseline is therefore taken from the earlier `dynabo_normal_scale` table, just as
    # the vanilla BO baseline above.
    dynabo_source = download_prior_tables("dynabo_sum", "iclr_rebuttal_prior_rf_yahpo", approach="dynabo")
    pibo_source = normalize_legacy_yahpo_costs(*download_prior_tables("dynabo_normal_scale", "iclr_rebuttal_prior_yahpo", approach="pibo"))
    prior_table, prior_incumbents, prior_priors = concat_prior_tables(dynabo_source, pibo_source)

    prior_table.to_csv(YAHPO_PRIOR_TABLE_PATH, index=False)
    prior_incumbents.to_csv(YAHPO_PRIOR_INCUMBENT_PATH, index=False)
    prior_priors.to_csv(YAHPO_PRIOR_PRIORS_PATH, index=False)


def download_mfpbench_rf_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_pd1")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_experiments")

    baseline_experimenter.get_table().to_csv(RF_PD1_BASELINE_TABLE_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(RF_PD1_BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(RF_PD1_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(RF_PD1_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(RF_PD1_PRIOR_PRIORS_PATH, index=False)


def download_mfpbench_gp_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_gp")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="pd1_prior_gaussian")

    baseline_experimenter.get_table().to_csv(GP_PD1_BASELINE_TABLE_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(GP_PD1_BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(GP_PD1_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(GP_PD1_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(GP_PD1_PRIOR_PRIORS_PATH, index=False)


def download_mfpbench_misleading_longer_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_pd1_longer")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_experiments_longer")

    baseline_experimenter.get_table().to_csv(RF_PD1_BASELINE_DECEIVING_LONGER_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(RF_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(RF_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(RF_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(RF_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH, index=False)

    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_gp_longer")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="pd1_prior_gaussian_longer")

    baseline_experimenter.get_table().to_csv(GP_PD1_BASELINE_DECEIVING_LONGER_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(GP_PD1_BASELINE_DECEIVING_LONGER_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(GP_PD1_DECEIVING_LONGER_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(GP_PD1_DECEIVING_LONGER_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(GP_PD1_DECEIVING_LONGER_PRIOR_PRIORS_PATH, index=False)


def download_dynamic_priors_data():
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="random_prior_location")

    prior_experimenter.get_table().to_csv(RF_PD1_DYNAMIC_PRIORS_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(RF_PD1_DYNAMIC_PRIORS_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(RF_PD1_DYNAMIC_PRIORS_PRIORS_PATH, index=False)


def download_remove_priors_ablation():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="remove_old_priors")

    ablation_experimenter.get_table().to_csv(REMOVE_OLD_PRIORS_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(REMOVE_OLD_PRIORS_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(REMOVE_OLD_PRIORS_ABLATION_PRIOR_PATH, index=False)


def download_mixed_priors():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="mixed_priors")

    ablation_experimenter.get_table().to_csv(MIXED_PRIORS_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(MIXED_PRIORS_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(MIXED_PRIORS_PRIORS_PATH, index=False)


def download_mfpbench_lcb_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_pd1_lcb")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_experiments_lcb")

    baseline_experimenter.get_table().to_csv(LCB_PD1_BASELINE_TABLE_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(LCB_PD1_BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(LCB_PD1_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(LCB_PD1_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(LCB_PD1_PRIOR_PRIORS_PATH, index=False)


def download_yahpo_lcb_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="baseline_yahpo_lcb")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_experiments_yahpo_lcb")

    baseline_experimenter.get_table().to_csv(LCB_YAHPO_BASELINE_TABLE_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(LCB_YAHPO_BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(LCB_YAHPO_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(LCB_YAHPO_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(LCB_YAHPO_PRIOR_PRIORS_PATH, index=False)


def download_prior_decay_ablation():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_decay_ablation")

    ablation_experimenter.get_table().to_csv(PRIOR_DECAY_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(PRIOR_DECAY_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(PRIOR_DECAY_ABLATION_PRIOR_PATH, index=False)


def download_n_rejection_samples_ablation():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_decay_n_rejection_sabmples_ablation")

    ablation_experimenter.get_table().to_csv(N_REJECTION_SAMPLES_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(N_REJECTION_SAMPLES_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(N_REJECTION_SAMPLES_ABLATION_PRIOR_PATH, index=False)


def download_prior_combination_ablation():
    ablation_experimenter = PyExperimenter(
        "dynabo/experiments/prior_combination_ablation/config.yml",
        CREDENTIALS_PATH,
        database_name="DynaBO_full_fidelity",
        table_name="prior_combination_ablation",
    )

    ablation_experimenter.get_table().to_csv(PRIOR_COMBINATION_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(PRIOR_COMBINATION_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(PRIOR_COMBINATION_ABLATION_PRIOR_PATH, index=False)


def download_prior_decay_ablation_yahpo():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_decay_ablation_yahpogym")

    ablation_experimenter.get_table().to_csv(YAHPO_PRIOR_DECAY_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(YAHPO_PRIOR_DECAY_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(YAHPO_PRIOR_DECAY_ABLATION_PRIOR_PATH, index=False)


if __name__ == "__main__":
    # download_mfpbench_rf_data()
    # download_mfpbench_gp_data()
    # download_mfpbench_misleading_longer_data()
    download_yahpo_data()
    # download_dynamic_priors_data()
    # download_remove_priors_ablation()
    # download_mixed_priors()
    # download_prior_decay_ablation()
    # download_mfpbench_lcb_data()
    # download_prior_decay_ablation_yahpo()
    # download_n_rejection_samples_ablation()
    # download_prior_combination_ablation()
    # download_yahpo_lcb_data()
