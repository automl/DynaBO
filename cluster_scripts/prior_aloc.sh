#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --array=1-250
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J dynabo-prior
#SBATCH -p normal
#SBATCH -A hpc-prf-intexml
#SBATCH -e /scratch/hpc-prf-intexml/fehring/DynaBO/logs/%x/%A_%a.err
#SBATCH -o /scratch/hpc-prf-intexml/fehring/DynaBO/logs/%x/%A_%a.out


cd /scratch/hpc-prf-intexml/fehring/DynaBO

conda init bash
ml lang
ml Miniforge3

source /pc2/users/i/intexml9/.bashrc
conda activate DynaBO

python dynabo/experiments/prior_experiments/execute_prior_experiments.py
