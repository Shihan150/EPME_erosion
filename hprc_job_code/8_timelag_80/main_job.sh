#!/bin/bash
#SBATCH --job-name=Early_PTB_Sr_temp_d13c_sens_sclim_pnas
#SBATCH --time=900:00
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=48
#SBATCH --mem=25G
#SBATCH --output=%x_%a.out


#SBATCH --mail-user=shihan@tamu.edu
#SBATCH --mail-type=END

# JOB HEADERS HERE

# some job information
echo "Starting at $(date)"
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition, the default partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs and ${mem_gbytes} GiB of RAM on compute node $(hostname)"

# echo the current zone
echo this is array job id $SLURM_ARRAY_JOB_ID
echo this is array task id $SLURM_ARRAY_TASK_ID

printf "\n\n"

module load Anaconda3/2024.02-1
# conda install python=3.9
python mcmc_temp_job_timelag_80.py
