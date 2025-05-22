#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --array=1-300
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J gt
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

python dynabo/experiments/gt_experiments/exectue_gt.py
