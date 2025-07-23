#!/bin/bash
#SBATCH --account=project_2013056
#SBATCH --ntasks=1
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module load pytorch
srun python3 -u /scratch/project_2013056/train.py
