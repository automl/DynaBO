from py_experimenter.experimenter import PyExperimenter

DATA_GENERATION_CONFIG_PATH = "dynabo/experiments/gt_experiments/config.yml"
BASELINE_CONFIG_PATH = "dynabo/experiments/baseline_experiments/config.yml"
PRIOR_EXPERIMENTS_PATH = "dynabo/experiments/prior_experiments/config.yml"
CREDENTIALS_PATH = "config/database_credentials.yml"

DATA_GENERATION_TABLE_PATH = "plotting_data/datageneration_table.csv"
DATA_GENERATION_INCUMBENT_PATH = "plotting_data/datageneration_incumbent.csv"
BASELINE_TABLE_PATH = "plotting_data/baseline_table.csv"
BASELINE_INCUMBENT_PATH = "plotting_data/baseline_incumbent.csv"
PRIOR_TABLE_PATH = "plotting_data/prior_table.csv"
PRIOR_INCUMBENT_PATH = "plotting_data/prior_incumbent.csv"
PRIOR_PRIORS_PATH = "plotting_data/prior_priors.csv"

if __name__ == "__main__":
    data_generation_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH)
    # baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH)
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH)

    data_generation_experimenter.get_table().to_csv(DATA_GENERATION_TABLE_PATH, index=False)
    data_generation_experimenter.get_logtable("incumbents").to_csv(DATA_GENERATION_INCUMBENT_PATH, index=False)

    # baseline_experimenter.get_table().to_csv(BASELINE_TABLE_PATH, index=False)
    # baseline_experimenter.get_logtable("incumbents").to_csv(BASELINE_INCUMBENT_PATH, index=False)

    prior_experimenter.get_table().to_csv(PRIOR_TABLE_PATH, index=False)
    prior_experimenter.get_logtable("incumbents").to_csv(PRIOR_INCUMBENT_PATH, index=False)
    prior_experimenter.get_logtable("priors").to_csv(PRIOR_PRIORS_PATH, index=False)
