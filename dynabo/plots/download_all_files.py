from py_experimenter.experimenter import PyExperimenter

baseline_config_path = "dynabo/experiments/baseline_experiments/config.yml"
prior_experiments_path = "dynabo/experiments/prior_experiments/config.yml"
credentials_path = "config/database_credentials.yml"

baseline_experimenter = PyExperimenter(baseline_config_path, credentials_path)
prior_experimenter = PyExperimenter(prior_experiments_path, credentials_path)

baseline_table = baseline_experimenter.get_table()
baseline_incumbent_df = baseline_experimenter.get_logtable("incumbents")

prior_table = prior_experimenter.get_table()
prior_incumbent_df = prior_experimenter.get_logtable("incumbents")
prior_priors_df = prior_experimenter.get_logtable("priors")

baseline_table.to_csv("dynabo/plots/baseline_table.csv", index=False)
baseline_incumbent_df.to_csv("dynabo/plots/baseline_incumbent.csv", index=False)
prior_table.to_csv("dynabo/plots/prior_table.csv", index=False)
prior_incumbent_df.to_csv("dynabo/plots/prior_incumbent.csv", index=False)
prior_priors_df.to_csv("dynabo/plots/prior_priors.csv", index=False)