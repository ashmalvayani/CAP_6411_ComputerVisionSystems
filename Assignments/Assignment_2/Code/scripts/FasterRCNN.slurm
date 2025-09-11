#!/bin/bash
#SBATCH --job-name=FasterRCNN
#SBATCH --error=outs/FasterRCNN-%J.err
#SBATCH --output=outs/FasterRCNN-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem24

conda activate /home/ashmal/anaconda3/envs/cvs_ass2

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

python fasterRCNN.py