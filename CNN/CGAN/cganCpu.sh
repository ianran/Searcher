#!/bin/bash
#SBATCH --job-name cganCPU
#SBATCH --output cganCPU.out
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --time 0-10:0:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --get-user-env

# Written Ian Rankin September 2018
# Slurm script for running the training onto.

echo "Loading modules"
# load needed modules
module load python-dev/2713
module load tensorflow-dev/CPU_only_1.8.0

echo "Starting execution of program."
# start running program

python -u ~/Searcher/CGAN/CGAN.py

echo "Completed execution"
