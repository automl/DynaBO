#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --array=1-4800
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 6G
#SBATCH -J dynabo-prior
#SBATCH -e /mnt/home/lfehring/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /mnt/home/lfehring/DynaBO/logs/%x/%A_%a.out


cd /mnt/home/lfehring/DynaBO

conda init bash
ml lang
ml Miniforge3

source /mnt/home/lfehring/.bashrc
conda activate DynaBO

/mnt/home/lfehring/DynaBO/.venv/bin/python dynabo/experiments/prior_experiments/execute_prior_experiments.py
