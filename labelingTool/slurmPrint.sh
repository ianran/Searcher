#!/bin/bash
#SBATCH --job-name zipCreation
#SBATCH --output zip.out
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 0-00:15:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --get-user-env

# Written Ian Rankin September 2018
# Slurm script for running the training onto.

echo "Started running zip creation"
# load needed modules
module load python/351
#module load tensorflow-dev/1.8.0

echo "Loaded module"
python /home/ianran/Searcher/labelingTool/print.py
