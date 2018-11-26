#!/bin/bash
#SBATCH --job-name extraction
#SBATCH --output extractImages.out
#SBATCH --partition normal
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --time 0-06:00:00
#SBATCH --mail-user ianran@nmsu.edu
#SBATCH --get-user-env

# extract.sh
# This files extracts images from every tar file in
# the scratch folder on ianran joker.

echo "Started extraction of images"

tar -xvzf $1

echo "Completed extraction"
