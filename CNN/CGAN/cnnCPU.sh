#!/bin/bash
#SBATCH --job-name trad
#SBATCH --output outCnnCPU.out
#SBATCH --partition normal
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

echo "/home/ianran/Searcher/labelingTool/FeedImagesToNumpyZip.py"

#sh ../../labelingTool/run.sh /scratch/ianran/validFeed/
#mv output.npz valid.npz

#sh ../../labelingTool/run.sh /home/ianran/feed/
#mv output.npz train.npz


module load tensorflow-dev/CPU_only_1.8.0

echo "Starting execution of program."
# start running program

python -u ~/Searcher/CNN/CGAN/TradCNN.py

echo "Completed execution"
