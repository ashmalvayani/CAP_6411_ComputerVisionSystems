#!/bin/bash
#SBATCH --job-name=SAM2
#SBATCH --error=jobs/SAM2-%J.err
#SBATCH --output=jobs/SAM2-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem24

conda activate /home/ashmal/anaconda3/envs/cvs_ass3

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

python sam2_experiment.py