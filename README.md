# DynaBO
This is the implementation of our Neurips 2025 submission titled **DynaBO: Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization**. In the paper we propose a method to incorporate dynamic user feedback in the form of priors at runtime.

![DynaBO evaluation results on lcbench](plots/scenario_plots/yahpogym/regret/lcbench.png)

## Install
To install and run our method, you need to execute the following steps:
1. Clone the repository with all additional dependencies using:
```bash
git clone --recursive https://github.com/automl/DynaBO.git 
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
Our experiments rely on the library. They therefore require either using a mysql or sqlite database. The process of using PyExperimenter is described in its [documentation](https://github.com/tornede/py_experimenter).
To replicate our experiments you need to execute the following steps
1. Create gt_data needed for priors by running: ``dynabo/experiments/gt_experiments/exectue_gt.py`` for both ``mfbench`` and ``yahpogym``. (As described in the paper, we executed one seed for one seed initially, and then only considered the learners classified as medium and hard.)
2. Create priors by running ``dynabo/data_processing/extract_gt_priors.py``
3. Execute the baselines, dynabo, and πBO using the scripts located in ``dynabo/experiments``. In our experiments ran slurm jobs utilizing the scripts in ``cluster_scripts``.