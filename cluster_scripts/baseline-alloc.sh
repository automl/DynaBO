#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --array=1-300
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J dynabo-basleine
#SBATCH -p normal
#SBATCH -A ####
#SBATCH -e /scratch/####/####/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /scratch/####/####/DynaBO/logs/%x/%A_%a.out


cd /scratch/####/####/DynaBO

conda init bash
ml lang
ml Miniforge3

source /####/####/i/####/.bashrc
conda activate DynaBO

python dynabo/experiments/baseline_experiments/execute_baseline.py
