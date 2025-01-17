#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH --array=1-832
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J dynabo-gt
#SBATCH -p normal
#SBATCH -A hpc-prf-intexml

cd /scratch/hpc-prf-intexml/wever/dynabo/DynaBO

pwd

ml lang
ml Miniforge3

source /pc2/users/w/wever/.bashrc

source activate dynabo

python -m dynabo.experiments.experimenter -e 1
