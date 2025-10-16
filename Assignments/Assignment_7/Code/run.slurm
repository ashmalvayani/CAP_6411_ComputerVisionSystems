#!/bin/bash
#SBATCH --job-name=SD35
#SBATCH --error=jobs/SD35-%J.err
#SBATCH --output=jobs/SD35-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass6

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

python sd3_5_prompts311_320.py