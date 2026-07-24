# DynaBO
This is the implementation accompanying our paper "Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization", **accepted at the AutoML Conference 2026**. In the paper, we propose a method to incorporate dynamic user feedback in the form of priors at runtime.

## Prerequisites

The project is built and installed entirely with [**uv**](https://docs.astral.sh/uv/), a fast Python package and environment manager. Our setup is firmly based on `uv`: it creates and manages the virtual environment and installs the correct Python interpreter for you.

You need the following on your system before installing:

| Requirement | Purpose | Install |
|---|---|---|
| `uv` (>= 0.4) | Creates the environment and resolves all dependencies from `uv.lock` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/) for other platforms) |
| `git` | Cloning the repository and its submodules | system package manager |
| `make` | Runs the install target | system package manager (`build-essential` on Debian/Ubuntu) |
| A C/C++ toolchain | Building some transitive dependencies (e.g. `swig`) | `build-essential` (Linux) / Xcode CLT (macOS) |

`uv` installs the correct Python interpreter (3.10) automatically, so no separate Python installation is required.

### Bundled dependencies (git submodules)

DynaBO depends on four repositories that are vendored as git submodules and pulled automatically during a recursive clone:

| Submodule | Path | Provides | Pinned commit |
|---|---|---|---|
| [CARP-S](https://github.com/automl/CARP-S) (`development` branch) | `CARP-S/` | Benchmark runner and the MFPBench benchmark integration | `b861cfc0483fc9bb8d9ad1779cf90ca6f1165531` |
| [SMAC3](https://github.com/automl/SMAC3) | `lib/SMAC3/` | The Bayesian optimization backend | `7f1ce0d1ba8536a052636b2f30929edcbca49e04` |
| [yahpo_gym](https://github.com/slds-lmu/yahpo_gym) | `lib/yahpo_gym/` | The YAHPO Gym benchmark library | `93f5b151d4e2f44daa5314cd10533aafec37d630` |
| [yahpo_data](https://github.com/slds-lmu/yahpo_data) | `benchmark_data/yahpo_data/` | Surrogate model data for YAHPO Gym | `efdab9072f63bd680396cd4b78b927c4a0caaad3` |

All submodule URLs are HTTPS, so no SSH key or GitHub account is required. A recursive clone checks out each submodule at the pinned commit listed above; if you obtain the submodules manually, check out these exact commits to reproduce our results.

## Install

1. Clone the repository **with all submodules**:
```bash
git clone --recursive https://github.com/automl/DynaBO.git
cd DynaBO
```
If you already cloned without `--recursive`, fetch the submodules with:
```bash
git submodule update --init --recursive
```
2. Install DynaBO and all dependencies into a uv-managed environment:
```bash
make install
```
This runs `uv sync` (resolving everything from `uv.lock`), patches the YAHPO Gym config-space files in `benchmark_data/yahpo_data`, and downloads the MFPBench (PD1) surrogate data into the environment. YAHPO Gym itself needs no additional download — its surrogate data is provided by the `benchmark_data/yahpo_data` submodule.

> **Reproducibility.** Exact dependency versions are not pinned in `pyproject.toml` (which only specifies compatible ranges); they are pinned in the committed `uv.lock`. Reproducing our environment therefore requires installing from the lockfile — `make install` does this via `uv sync`, which installs the exact locked versions (with hashes) rather than re-resolving. Do not run `uv sync --upgrade` or delete `uv.lock` if you want to match the versions we used.

3. Activate the environment that `uv` created, then run commands with plain `python`:
```bash
source .venv/bin/activate
python examples/baseline/example.py
```

> **Run from the repository root.** YAHPO Gym is located via the relative path `benchmark_data/yahpo_data`, so all scripts and experiments must be launched from the repository root.

## Quick start: run the minimal examples

If you only want to confirm the code runs end-to-end (the fastest path to a partial reproduction), use the self-contained examples in [`examples/`](examples/). They use MFPBench and log to a local SQLite database — **no MySQL server or database credentials are needed**. See the [Minimal Examples](#minimal-examples) section below.

## Execution
Our experiments rely on the PyExperimenter library. You can run a local version with SQLite, but for large-scale experiments and reproducing the results, we suggest setting up a MySQL database server.
The process of using PyExperimenter is described in its [documentation](https://github.com/tornede/py_experimenter).

To replicate our experiments, you need to execute the following steps:
1. Create gt_data needed for priors by running: ``dynabo/experiments/data_generation/execute_baseline.py`` for both ``mfpbench`` and ``yahpogym``. 
    We did this with both expected improvement and confidence bound acquisition functions.
2. Create priors by running ``dynabo/data_processing/cluster_incumbents.py``
    This will extract the entries from the database, cluster them, and save the priors to disk. To replicate the PC results, you need to either copy the files over or link the path.
3. Execute the baselines, DynaBO, and πBO using the scripts located in ``dynabo/experiments``. In our experiments, we ran Slurm jobs utilizing the scripts in ``cluster_scripts`` but parallelization requires a MySQL database server.
    This will populate the database with entries and continuously pull and execute experiments.
4. Download the results from the database using ``dynabo/data_processing/download_all_files.py``
5. Create plots in ``dynabo/plotting``.

### Experiments
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




## Minimal Examples

Three self-contained examples are provided in `examples/`. Each logs results to a local SQLite database — no MySQL server or credentials file required. The baseline and DynaBO examples use MFPBench (`lm1b_transformer_2048`); the YAHPO example uses YAHPO Gym (`lcbench`), whose surrogate data comes from the `benchmark_data/yahpo_data` submodule (no additional download).

| Example | Script | Config | SQLite database |
|---|---|---|---|
| Baseline on MFPBench (plain SMAC) | `examples/baseline/example.py` | `examples/baseline/config.yml` | `examples/baseline/baseline.db` |
| Baseline on YAHPO Gym (plain SMAC) | `examples/yahpo/example.py` | `examples/yahpo/config.yml` | `examples/yahpo/yahpo.db` |
| DynaBO (dynamic priors) | `examples/dynabo/example.py` | `examples/dynabo/config.yml` | `examples/dynabo/dynabo.db` |

Run from the repository root with the environment activated (`source .venv/bin/activate`):

```bash
python examples/baseline/example.py
python examples/yahpo/example.py
python examples/dynabo/example.py
```

Each script fills the database with one experiment configuration and executes it. Results (final cost, runtime) are written to the SQLite database on completion. For the YAHPO example the objective is validation accuracy, so `final_cost` is stored as its negation (SMAC minimizes). The DynaBO example additionally logs per-trial incumbent trajectories and prior injection events to the `configs` and `priors` logtables.

> **Note:** The DynaBO example requires prior data to be present under `benchmark_data/prior_data/` (generated via step 2 of the Execution instructions above). Because this data may not be available in all setups, the result of one completed run is already stored in `examples/dynabo/dynabo.db` so the output format can be inspected without re-running the experiment.

Results can be inspected with any SQLite client, e.g.:

```bash
sqlite3 examples/dynabo/dynabo.db "SELECT * FROM dynabo_runs;"
sqlite3 examples/dynabo/dynabo.db "SELECT * FROM dynabo_runs__configs;"
sqlite3 examples/dynabo/dynabo.db "SELECT * FROM dynabo_runs__priors;"
```

## Comparison to "Hyperparameter Optimization via Interacting with Probabilistic Circuits"

For a comparison with [Probabilistic Circuits](https://github.com/ml-research/ibo-hpc) we utilize a [forked version of their repository](https://github.com/LUH-AI/ibo-hpc).
After execution, you need to copy the results from their repository to `dynabo/plotting_data/pc_results`. 
