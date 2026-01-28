# DynaBO
This is the implementation of our submission titled DynaBO: Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization. In the paper, we propose a method to incorporate dynamic user feedback in the form of priors at runtime.

## Install
To install and run our method, you need to execute the following steps:
1. Clone the repository with all additional dependencies using:
```bash
git clone --recursive https://github.com/OrgName/DynaBO.git 
```
2. Create a conda environment and activate it using:
```bash
conda create -n DynaBO python=3.10
conda activate DynaBO
```
3. Install the repository and all dependencies:
```bash
make install
```

## Execution
Our experiments rely on the PyExperimenter library. You can run a local version with SQLite, but for large-scale experiments and reproducing the results, we suggest setting up a MySQL database server.
The process of using PyExperimenter is described in its [documentation](https://github.com/tornede/py_experimenter).

To replicate our experiments, you need to execute the following steps:
1. Create gt_data needed for priors by running: ``dynabo/experiments/data_generation/execute_baseline.py`` for both ``mfbench`` and ``yahpogym``. 
    We did this with both expected improvement and confidence bound acquisition functions.
2. Create priors by running ``dynabo/data_processing/extract_gt_priors.py``
    This will extract the entries from the database, cluster them, and save the priors to disk. To replicate the PC results, you need to either copy the files over or link the path.
3. Execute the baselines, DynaBO, and πBO using the scripts located in ``dynabo/experiments``. In our experiments, we ran Slurm jobs utilizing the scripts in ``cluster_scripts`` but parallelization requires a MySQL database server.
    This will populate the database with entries and continuously pull and execute experiments.
4. Download the results from the database using ``dynabo/data_processing/download_all_files.py``
5. Create plots in ``dynabo/plotting``.

### Structure of Experiemnts
Every experiment is located in ``dynabo/experiments/``, and contains both a config file and a Python file. The structure of the config files is described in the [PyExperimenter documentation](https://github.com/tornede/py_experimenter).

The python file is structured as follows 

```python

...

def run_experiment(config: dict, result_processor: ResultProcessor, custom_cfg: dict):
    # Some target function

    result = {
        "initial_design_size": initial_design_size,
        "final_cost": optimization_data["final_cost"],
        "runtime": round(end_time - start_time, 3),
        "virtual_runtime": optimization_data["virtual_runtime"],
        "reasoning_runtime": round(evaluator.reasoning_runtime, 3),
        "n_evaluations_computed": optimization_data["n_evaluations_computed"],
        "experiment_finished": True,
    }

    result_processor.process_results(results=result)


if __name__ == "__main__":
    ...
    experimenter = PyExperimenter(  # Creation of the experimenter
        experiment_configuration_file_path=EXP_CONFIG_FILE_PATH,  # Path to the config file
        database_credential_file_path=DB_CRED_FILE_PATH,  # Path to the database credentials; not needed for SQLite
        use_codecarbon=False,
    )

    # Information to fill the database
    fill = True  # Whether to fill the database with experiments
    benchmarklib = "mfbench"  # Benchmark library
    if fill:
        fill_table(
            py_experimenter=experimenter,
            common_parameters={  # General setup parameters
                "acquisition_function": ["expected_improvement"],
                "timeout_total": [3600],
                "n_trials": [500],
                "initial_design__n_configs_per_hyperparameter": [10],
                "initial_design__max_ratio": [0.25],
                "seed": list(range(30)),
            },
            benchmarklib=benchmarklib,  # Benchmark library to use
            benchmark_parameters={  # Benchmark-specific parameters
                "with_all_datasets": True,
                "medium_and_hard": False,
            },
            approach="baseline",
            approach_parameters=None,
        )

    # Whether to reset experiments with status error or running
    reset = False
    if reset:
        experimenter.reset_experiments("error", "running")

    # Execute experiments
    execute = True
    if execute:
        experimenter.execute(run_experiment, max_experiments=1, random_order=True)

```




## Comparison to "Hyperparameter Optimization via Interacting with Probabilistic Circuits"

For a comparison with [Probabilistic Circuits](https://github.com/ml-research/ibo-hpc) we utilize a forked version of their repository. You can find it https://anonymous.4open.science/r/ibo-hpc-7C28/README.md
After execution, you need to copy the results from their repository to `dynabo/plotting_data/pc_results`. 
