#!/bin/bash

# PBS directives
#---------------
#PBS -N gridsearch
#PBS -l nodes=1:ncpus=16
#PBS -l walltime=24:00:00
#PBS -q one_day
#PBS -m abe
#PBS -M shangda.zhu@cranfield.ac.uk

# Output and error join
#PBS -j oe
# Disable GPU visibility if not needed
#PBS -v "CUDA_VISIBLE_DEVICES="
# Use private sandbox
#PBS -W sandbox=PRIVATE
# Don't keep unfinished job files
#PBS -k n

# Link job dir (optional)
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID

# Move to job working directory
cd $PBS_O_WORKDIR

# Count CPUs
export cpus=$(cat $PBS_NODEFILE | wc -l)

# Load Anaconda
module use /apps2/modules/all
module load Anaconda3

# Activate conda env
source /apps2/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate graphdta_env  # 改为你实际创建的环境名

# Run grid search
python grid_search.py

# Clean up logs
rm $PBS_O_WORKDIR/$PBS_JOBID
