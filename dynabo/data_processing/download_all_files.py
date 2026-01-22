from py_experimenter.experimenter import PyExperimenter

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


def download_yahpo_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, database_name="dynabo_normal_scale", table_name="baseline_yahpo")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="dynabo_sum", table_name="iclr_rebuttal_prior_rf_yahpo")

    baseline_experimenter.get_table().to_csv(YAHPO_BASELINE_TABLE_PATH, index=False)
    baseline_experimenter.get_logtable("configs").to_csv(YAHPO_BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(YAHPO_PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(YAHPO_PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(YAHPO_PRIOR_PRIORS_PATH, index=False)


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


def download_prior_decay_ablation():
    ablation_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, database_name="DynaBO_full_fidelity", table_name="prior_decay_ablation")

    ablation_experimenter.get_table().to_csv(PRIOR_DECAY_ABLATION_TABLE_PATH, index=False)
    ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(PRIOR_DECAY_ABLATION_INCUMBENT_PATH, index=False)
    ablation_experimenter.get_logtable("priors").to_csv(PRIOR_DECAY_ABLATION_PRIOR_PATH, index=False)


if __name__ == "__main__":
    # download_mfpbench_rf_data()
    # download_mfpbench_gp_data()
    # download_mfpbench_misleading_longer_data()
    # download_yahpo_data()
    # download_dynamic_priors_data()
    download_remove_priors_ablation()
    # download_mixed_priors()
    # download_prior_decay_ablation()
