#!/bin/bash
#SBATCH --job-name=SDXL
#SBATCH --error=jobs/SDXL-%J.err
#SBATCH --output=jobs/SDXL-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass6

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

python sdxl_eval_prompts311_320.py