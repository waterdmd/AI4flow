#!/bin/bash
#SBATCH -p general
#SBATCH -N 1            # number of nodes
#SBATCH -c 16           # number of cores 
#SBATCH --gres=gpu:a100:1    # request 1 GPU
#SBATCH -t 0-24:00:00   # time in d-hh:mm:ss
#SBATCH --mem=128G      # memory for all cores in GB
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)

# Load required modules for job's environment
source activate tf-gpu

# Run the Python script with the specific config file passed as an argument
python /scratch/kdahal3/camels_losses/main.py --config $1