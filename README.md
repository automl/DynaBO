# DynaBO


## Install

1. First create a `conda` environment and activate it using 
```bash
conda create -n DynaBO python-3.10
conda activate DynaBO
```

2. Clone SMAC and install from fix-priorAcquisitionFunction branch (TODO this should be merged before the release of our codebase)
```bash
git clone https://github.com/automl/SMAC3.git
cd SMAC3
git checkout 1076-fix-priorAcquisitionFunction
pip install .
```

3. Install our project in an editable state
```bash
pip install -e .
```