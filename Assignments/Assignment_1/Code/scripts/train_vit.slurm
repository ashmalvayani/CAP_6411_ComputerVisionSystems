#!/bin/bash
#SBATCH --job-name=train-ViT
#SBATCH --error=outs/train-ViT-%J.err
#SBATCH --output=outs/train-ViT-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass1

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

python train_vit.py