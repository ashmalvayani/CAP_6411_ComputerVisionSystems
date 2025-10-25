#!/bin/bash
#SBATCH --job-name=WAN
#SBATCH --error=jobs/WAN-%J.err
#SBATCH --output=jobs/WAN-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:2
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/8

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

cd diffusion-pipe
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=2 train.py --deepspeed --config examples/wan_har_lora.toml