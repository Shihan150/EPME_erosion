#!/bin/bash
#SBATCH --job-name=Early_PTB_10kyr_P_on_erosion_sclim                                 
#SBATCH --time=1200:00                                            
#SBATCH --ntasks=1                                             

#SBATCH --cpus-per-task=25      
#SBATCH --mem=25G
#SBATCH --output=%x_%a.out


#SBATCH --mail-user=shihan@tamu.edu
#SBATCH --mail-type=END    
#SBATCH --array=2,6
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

module load Anaconda3/2022.10
conda install python=3.9
python mcmc_job_nsi_${SLURM_ARRAY_TASK_ID}.py

