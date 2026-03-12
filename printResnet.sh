#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=PRresnet50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=print_resnet50.log
#SBATCH --partition=eternity

#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "Experiment 1: Src=0% (0 samples), Tgt=100% (10 samples)"
echo "----------------------------------------------------"
# dont SBATCH --gres=gpu:4

export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

echo "Starting Unit Test..."

python printModel.py

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
