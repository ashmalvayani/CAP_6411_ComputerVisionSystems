#!/bin/bash
#SBATCH --job-name=DINO
#SBATCH --error=outs/DINO-%J.err
#SBATCH --output=outs/DINO-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem24

conda activate /home/ashmal/anaconda3/envs/cvs_ass2

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

cd DINO

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
bash scripts/DINO_eval.sh /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017 Weights/checkpoint0011_4scale.pth