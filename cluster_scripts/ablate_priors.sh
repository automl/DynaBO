#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --array=1-300
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 6G
#SBATCH -J dynabo-prior
#SBATCH -e /mnt/home/username/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /mnt/home/username/DynaBO/logs/%x/%A_%a.out


cd /mnt/home/username/DynaBO

sleep $(( RANDOM % 200 ))

conda init bash
ml lang
ml Miniforge3

source /mnt/home/username/.bashrc
conda activate DynaBO

/mnt/home/username/DynaBO/.venv/bin/python dynabo/experiments/ablate_all_priors/ablate_priors_experiements.py