#!/bin/bash
#SBATCH --job-name cgan1
#SBATCH --output cgan2.out
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --time 2-00:00:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --get-user-env

# Written Ian Rankin September 2018
# Slurm script for running the training onto.

echo "Loading modules"
# load needed modules
module load python-dev/361
module load tensorflow-dev/1.8.0

echo "Starting execution of program."
# start running program

python -u ~/Searcher/CNN/CGAN/CGANRealData.py

echo "Completed execution"
