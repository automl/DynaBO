#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --array=1-120
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 6G
#SBATCH -J baseline-alloc
#SBATCH -e /mnt/home/lfehring/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /mnt/home/lfehring/DynaBO/logs/%x/%A_%a.out

cd /mnt/home/lfehring/DynaBO

conda init bash
ml lang
ml Miniforge3

source /mnt/home/lfehring/.bashrc
conda activate DynaBO

/mnt/home/lfehring/DynaBO/.venv/bin/python dynabo/experiments/baseline_experiments/execute_baseline.py
