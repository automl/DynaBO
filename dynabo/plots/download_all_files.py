from py_experimenter.experimenter import PyExperimenter

baseline_config_path = "dynabo/experiments/baseline_experiments/config.yml"
dynabo_config_path = "dynabo/experiments/dynabo_experiments/config.yml"
pibo_config_path = "dynabo/experiments/pibo_experiments/config.yml"
credentials_path = "config/database_credentials.yml"

baseline_experimenter = PyExperimenter(baseline_config_path, credentials_path)
dynabo_experimenter = PyExperimenter(dynabo_config_path, credentials_path)
pibo_experimenter = PyExperimenter(pibo_config_path, credentials_path)

baseline_table = baseline_experimenter.get_table()
baseline_incumbent_df = baseline_experimenter.get_logtable("incumbents")

dynabo_table = dynabo_experimenter.get_table()
dynabo_incumbent_df = dynabo_experimenter.get_logtable("incumbents")
dynabo_priors_df = dynabo_experimenter.get_logtable("priors")

pibo_table = pibo_experimenter.get_table()
pibo_incumbent_df = pibo_experimenter.get_logtable("incumbents")
pibo_priors_df = pibo_experimenter.get_logtable("priors")

baseline_table.to_csv("dynabo/plots/baseline_table.csv", index=False)
baseline_incumbent_df.to_csv("dynabo/plots/baseline_incumbent.csv", index=False)
dynabo_table.to_csv("dynabo/plots/prior_table.csv", index=False)
dynabo_incumbent_df.to_csv("dynabo/plots/prior_incumbent.csv", index=False)
dynabo_priors_df.to_csv("dynabo/plots/prior_priors.csv", index=False)
pibo_table.to_csv("dynabo/plots/pibo_table.csv", index=False)
pibo_incumbent_df.to_csv("dynabo/plots/pibo_incumbent.csv", index=False)
pibo_priors_df.to_csv("dynabo/plots/pibo_priors.csv", index=False)
