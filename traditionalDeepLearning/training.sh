#!/bin/bash
#SBATCH --job-name trad_training
#SBATCH --output trad_training.out
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 0-00:15:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --get-user-env

# Written Ian Rankin September 2018
# Slurm script for running the training onto.

echo "Loading modules"
# load needed modules
module load python/351
module load tensorflow-dev/1.8.0

echo "Starting execution of program."
# start running program

python -u ~/Searcher/traditionalDeepLearning/ClassifyCNN.py

echo "Completed execution"
