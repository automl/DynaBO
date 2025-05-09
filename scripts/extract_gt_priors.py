import os

from dynabo.utils.data_utils import build_prior_dataframe, create_prior_data_path_pd1, create_prior_data_path_yahpo, load_df, save_base_table, save_table


def extract_gt_priors_pd1():
    """
    This function creates the prior data for the pd1 benchmark.
    """
    save_base_table("mfpbench", "data_generation_pd1")
    base_df = load_df(benchmark_name="mfpbench")
    for scenario in base_df.scenario.unique():
        scenario_df = base_df[base_df.scenario == scenario]
        prior_data_path = create_prior_data_path_pd1(scenario)
        prior_df = build_prior_dataframe(scenario_df, filter_with_epsilon_distance=True, filter_epsilon=0.005)
        save_table(prior_df, os.path.join(prior_data_path, "prior_table.csv"))
        print(f"Finished scenario: {scenario}")


def extract_gt_priors_yahpo():
    """
    This function creates the prior data for the yahpo benchmark.
    """
    save_base_table("yahpo", "data_generation_medium_hard_new")
    base_df = load_df(benchmark_name="yahpo")
    for scenario in base_df.scenario.unique():
        scenario_df = base_df[base_df.scenario == scenario]
        for dataset in scenario_df.dataset.unique():
            dataset_df = scenario_df[scenario_df.dataset == dataset]
            for metric in dataset_df.metric.unique():
                metric_df = dataset_df[dataset_df.metric == metric]
                prior_data_path = create_prior_data_path_yahpo(scenario, dataset, metric)
                prior_df = build_prior_dataframe(metric_df, filter_with_epsilon_distance=True, filter_epsilon=0.005)
                save_table(prior_df, os.path.join(prior_data_path, "prior_table.csv"))
            print(f"Finished scenario: {scenario}, dataset: {dataset}, metric: {metric}")


if __name__ == "__main__":
    extract_gt_priors_pd1()
    extract_gt_priors_yahpo()
