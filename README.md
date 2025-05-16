# DynaBO


## Install

1. Fist clone the repository with all additional dependencies using:
```bash
git clone ---recursives https://github.com/automl/DynaBO.git 
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
