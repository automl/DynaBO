# DynaBO
This is the implementation of our Neurips 2025 submission titled **DynaBO: Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization**. In the paper we propose a method to incorporate dynamic user feedback in the form of priors at runtime.

## Install
To install and run our method, you need to execute the following steps
1. Fist clone the repository with all additional dependencies using:
```bash
git clone --recursive https://github.com/automl/DynaBO.git 
```
2. First create a `conda` environment and activate it using:
```bash
conda create -n DynaBO python=3.10
conda activate DynaBO
```
3. Install the repo and all dependencies:
```bash
make install
```
4. Enable using mfpbench in CARP-S
```bash
cd CARP-S
make benchmark_mfpbench
```
