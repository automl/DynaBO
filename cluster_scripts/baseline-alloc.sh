#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --array=1-80
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G
#SBATCH -J baseline-alloc
#SBATCH -e /mnt/home/username/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /mnt/home/username/DynaBO/logs/%x/%A_%a.out

cd /mnt/home/username/DynaBO

conda init bash
ml lang
ml Miniforge3

source /mnt/home/username/.bashrc
conda activate DynaBO

/mnt/home/username/DynaBO/.venv/bin/python dynabo/experiments/data_generation/execute_baseline.py