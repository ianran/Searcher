#!/bin/bash
#SBATCH --job-name zipCreation
#SBATCH --output zipValid.out
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --time 0-06:00:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --get-user-env

# Written Ian Rankin September 2018
# Slurm script for running the training onto.

echo "Started running zip creation"
# load needed modules
module load python/351
#module load tensorflow-dev/1.8.0

echo "Loaded module"
# start running program
echo "/home/ianran/Searcher/labelingTool/FeedImagesToNumpyZip.py"

#python /home/ianran/Searcher/labelingTool/FeedImagesToNumpyZip.py /home/ianran/feed/

#sh t.sh

sh run.sh /scratch/ianran/validFeed/

#python /home/ianran/Searcher/labelingTool/FeedImagesToNumpyZip.py /home/ianran/feed/ /home/ianran/Searcher/labelingTool/labels.csv 1

echo "Finished running program"
