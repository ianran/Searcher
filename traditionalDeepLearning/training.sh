#!/bin/bash
#SBATCH --job-name trad_training
#SBATCH --output ~/trad_training.out
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 0-00:15:00
#SBATCH --mail-user ianran@nmsu.edu
##SBATCH --mail-type BEGIN
##SBATCH --mail-type END
##SBATCH --mail-type FAIL

# Written Ian Rankin September 2018
# Slurm script for running the training onto.


# load needed modules
module load python/351
module load tensorflow-dev/1.8.0

# start running program
python ~/Searcher/traditionalDeepLearning/ClassifyCNN.py
