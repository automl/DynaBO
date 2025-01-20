import os

from dynabo.utils.data_utils import build_prior_dataframe, create_prior_data_path, load_df, save_table, save_base_table

if __name__ == "__main__":
    save_base_table()

    base_df = load_df()
    base_df = base_df[base_df.status == "done"]
    for scenario in base_df.scenario.unique():
        scenario_df = base_df[base_df.scenario == scenario]

        for dataset in scenario_df.dataset.unique():
            dataset_df = scenario_df[scenario_df.dataset == dataset]

            for metric in dataset_df.metric.unique():
                metric_df = dataset_df[dataset_df.metric == metric]

                prior_data_path = create_prior_data_path(scenario, dataset, metric)

                prior_df = build_prior_dataframe(metric_df)
                save_table(prior_df, os.path.join(prior_data_path, "prior_table.csv"))
