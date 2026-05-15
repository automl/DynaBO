#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 32G
#SBATCH -J cluster_incumbents
#SBATCH -e /mnt/home/username/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /mnt/home/username/DynaBO/logs/%x/%A_%a.out


cd /mnt/home/username/DynaBO

conda init bash
ml lang
ml Miniforge3

source /mnt/home/username/.bashrc
conda activate DynaBO

/mnt/home/username/DynaBO/.venv/bin/python dynabo/data_processing/cluster_incumbents.py