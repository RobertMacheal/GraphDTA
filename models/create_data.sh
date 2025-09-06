#!/bin/bash


# PBS directives
#---------------

#PBS -N create_data
#PBS -l nodes=1:ncpus=16
#PBS -l walltime=03:00:00
#PBS -q three_hour
#PBS -m abe
#PBS -M shangda.zhu@cranfield.ac.uk

#===============
#PBS -j oe
#PBS -v "CUDA_VISIBLE_DEVICES="
#PBS -W sandbox=PRIVATE
#PBS -k n
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
## Change to working directory
cd $PBS_O_WORKDIR
## Calculate number of CPUs and GPUs
export cpus=`cat $PBS_NODEFILE | wc -l`
## Load production modules
module use /apps2/modules/all
## =============


# Stop at runtime errors
set -e
# Load required modules
module load Anaconda3

source /apps2/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

conda activate graphdta_env

python create_data.py



## Tidy up the log directory
## =========================
rm $PBS_O_WORKDIR/$PBS_JOBID
