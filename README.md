# DynaBO
This is the implementation of our submission titled **DynaBO: Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization**. In the paper we propose a method to incorporate dynamic user feedback in the form of priors at runtime.

## Install
To install and run our method, you need to execute the following steps:
1. Clone the repository with all additional dependencies using:
```bash
git clone --recursive https://github.com/OrgName/DynaBO.git 
```
2. Create a `conda` environment and activate it using:
```bash
conda create -n DynaBO python=3.10
conda activate DynaBO
```
3. Install the repo and all dependencies:
```bash
make install
```

## Execution
Our experiments rely on the PyExperimenter library. You can run a local version with SQLite, buty for large scale experiments, reproducing the results we suggest setting up a mysql database server. 
The process of using PyExperimenter is described in its [documentation](https://github.com/tornede/py_experimenter).

To replicate our experiments you need to execute the following steps
1. Create gt_data needed for priors by running: ``dynabo/experiments/data_generation/execute_baseline.py`` for both ``mfbench`` and ``yahpogym``. 
    We did this with both expected improvement and confidence bound acquisition functions.
2. Create priors by running ``dynabo/data_processing/extract_gt_priors.py``
    This will extract the entries from the database, cluster them, and save the priors to disk. To replicate the pc results, you need to either copy the files over, or link the path.
3. Execute the baselines, dynabo, and πBO using the scripts located in ``dynabo/experiments``. In our experiments ran slurm jobs utilizing the scripts in ``cluster_scripts`` bur parallelisation requires a mysql database server.
    This will populate the database with entried, and continiously pull experiments and execute them.
4. Download the results from the database using ``dynabo/data_processing/download_all_files.py``
4. Create plots in ``dynabo/plotting``.

## Comparison to "Hyperparameter Optimization via Interacting with Probabilistic Circuits"

For a comparison with [Probabilistic Circuits](https://github.com/ml-research/ibo-hpc) we utilize a forked version of their repository. You can find it https://anonymous.4open.science/r/ibo-hpc-7C28/README.md
After execution, you need to copy the results from their repository to `dynabo/plotting_data/pc_results`. 
