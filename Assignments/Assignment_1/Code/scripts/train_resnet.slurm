#!/bin/bash
#SBATCH --job-name=train-ResNet18
#SBATCH --error=outs/train-ResNet18-%J.err
#SBATCH --output=outs/train-ResNet18-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass1

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

python train_resnet.py