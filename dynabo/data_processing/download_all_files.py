from py_experimenter.experimenter import PyExperimenter

DATA_GENERATION_CONFIG_PATH = "dynabo/experiments/gt_experiments/config.yml"
BASELINE_CONFIG_PATH = "dynabo/experiments/baseline_experiments/config.yml"
PRIOR_EXPERIMENTS_PATH = "dynabo/experiments/prior_experiments/config.yml"
CREDENTIALS_PATH = "config/database_credentials.yml"

YHAPO_DATA_GENERATION_ONE_SEED_PATH = "plotting_data/yahpo/datageneration.csv"
YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH = "plotting_data/yahpo/datageneration_incumbent.csv"

YAHPO_DATA_GENERATION_MEDIUM_HARD_PATH = "plotting_data/yahpo/datageneration_medium_hard.csv"
YAHPO_DATA_GENERATION_INCUMBENT_MEDIUM_HARD_PATH = "plotting_data/yahpo/datageneration_incumbent_medium_hard.csv"

YAHPO_BASELINE_TABLE_PATH = "plotting_data/yahpo/baseline.csv"
YAHPO_BASELINE_INCUMBENT_PATH = "plotting_data/yahpo/baseline_incumbent.csv"
YAHPO_PRIOR_TABLE_PATH = "plotting_data/yahpo/prior.csv"
YAHPO_PRIOR_INCUMBENT_PATH = "plotting_data/yahpo/prior_incumbent.csv"
YAHPO_PRIOR_PRIORS_PATH = "plotting_data/yahpo/prior_priors.csv"

if __name__ == "__main__":
    # Download all Files for yahpogym
    data_generation_one_seed_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="data_generation_new")
    data_generation_medium_hard_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="data_generation_medium_hard_new")
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH)
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, table_name="prior_approaches_new")
    prior_experimenter_with_disregarding = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, table_name="dynabo")

    try:
        data_generation_one_seed_experimenter.get_table().to_csv(YHAPO_DATA_GENERATION_ONE_SEED_PATH, index=False)
        data_generation_one_seed_experimenter.get_logtable("configs").to_csv(YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH, index=False)
    except Exception:
        print("No DataGeneration One Seed")

    try:
        data_generation_medium_hard_experimenter.get_table().to_csv(YAHPO_DATA_GENERATION_MEDIUM_HARD_PATH, index=False)
        data_generation_medium_hard_experimenter.get_logtable("configs").to_csv(YAHPO_DATA_GENERATION_INCUMBENT_MEDIUM_HARD_PATH, index=False)
    except Exception:
        print("No DataGeneration Medium Hard")

    try:
        baseline_experimenter.get_table().to_csv(YAHPO_BASELINE_TABLE_PATH, index=False)
        baseline_experimenter.get_logtable("configs").to_csv(YAHPO_BASELINE_INCUMBENT_PATH, index=False)
    except Exception:
        print("No Baseline")

    try:
        prior_experimenter.get_table().to_csv(YAHPO_PRIOR_TABLE_PATH, index=False)
        prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(YAHPO_PRIOR_INCUMBENT_PATH, index=False)
        prior_experimenter.get_logtable("priors").to_csv(YAHPO_PRIOR_PRIORS_PATH, index=False)
    except Exception:
        print("No Prior")

    # Download all files for pd1
    data_generation_pd1_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="data_generation_pd1")

    try:
        data_generation_pd1_experimenter.get_table().to_csv(YHAPO_DATA_GENERATION_ONE_SEED_PATH, index=False)
        data_generation_pd1_experimenter.get_logtable("configs").to_csv(YAHPO_DATA_GENERATION_INCUMBENT_ONE_SEED_PATH, index=False)
    except Exception:
        print("No DataGeneration data pd1")
