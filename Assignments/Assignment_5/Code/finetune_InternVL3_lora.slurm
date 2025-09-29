#!/bin/bash
#SBATCH --job-name=InternVL3
#SBATCH --error=jobs/InternVL3-%J.err
#SBATCH --output=jobs/InternVL3-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:2
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass5_internvl

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

cd InternVL/internvl_chat
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/lora_internvl3_8b.sh