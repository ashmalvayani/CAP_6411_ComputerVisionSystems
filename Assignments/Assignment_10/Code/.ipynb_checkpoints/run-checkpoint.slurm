#!/bin/bash
#SBATCH --job-name=TDM
#SBATCH --error=jobs/TDM-%J.err
#SBATCH --output=jobs/TDM-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:turing:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmemT48

conda activate /home/ashmal/anaconda3/envs/cvs_ass10

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

cd TDM
bash run.sh