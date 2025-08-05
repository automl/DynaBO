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

PD1_BASELINE_TABLE_PATH = "plotting_data/pd1/baseline.csv"
PD1_BASELINE_INCUMBENT_PATH = "plotting_data/pd1/baseline_incumbent.csv"
PD1_PRIOR_TABLE_PATH = "plotting_data/pd1/prior.csv"
PD1_PRIOR_INCUMBENT_PATH = "plotting_data/pd1/prior_incumbent.csv"
PD1_PRIOR_PRIORS_PATH = "plotting_data/pd1/prior_priors.csv"


def download_yahpo_data():
    data_generation_one_seed_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="data_generation_new")
    data_generation_medium_hard_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="data_generation_medium_hard_new")
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, table_name="baseline_new")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, table_name="prior_approaches_new")
    yahpo_ablation_experimenter = PyExperimenter(DATA_GENERATION_CONFIG_PATH, CREDENTIALS_PATH, table_name="dynabo_ablation_fix")

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

    try:
        # Remove error columns because it can cause issues for a very large table
        table = yahpo_ablation_experimenter.get_table()
        table = table.drop(
            columns=[
                "error",
            ],
        )
        table.to_csv(YAHPO_ABLATION_TABLE_PATH, index=False)

        yahpo_ablation_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(YAHPO_ABLATION_INCUMBENT_PATH, index=False)
        yahpo_ablation_experimenter.get_logtable("priors").to_csv("plotting_data/yahpogym/yahpo_ablation_priors.csv", index=False)
    except Exception:
        print("No Yahpo Ablation")


def download_mfpbench_data():
    baseline_experimenter = PyExperimenter(BASELINE_CONFIG_PATH, CREDENTIALS_PATH, table_name="baseline")
    prior_experimenter = PyExperimenter(PRIOR_EXPERIMENTS_PATH, CREDENTIALS_PATH, table_name="dynabo_random_prior_location")

    try:
        baseline_experimenter.get_table().to_csv(PD1_BASELINE_TABLE_PATH, index=False)
        baseline_experimenter.get_logtable("configs").to_csv(PD1_BASELINE_INCUMBENT_PATH, index=False)
    except Exception:
        print("No Baseline")

    try:
        prior_experimenter.get_table().to_csv(PD1_PRIOR_TABLE_PATH, index=False)
        prior_experimenter.get_logtable("configs", condition="incumbent = 1").to_csv(PD1_PRIOR_INCUMBENT_PATH, index=False)
        prior_experimenter.get_logtable("priors").to_csv(PD1_PRIOR_PRIORS_PATH, index=False)
    except Exception:
        print("No Prior")


if __name__ == "__main__":
    download_mfpbench_data()
    # download_yahpo_data()
