#!/bin/bash
#SBATCH --job-name=SophiaVL-R1
#SBATCH --error=jobs/SophiaVL-R1-%J.err
#SBATCH --output=jobs/SophiaVL-R1-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:2
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass5_qwen25vl

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

cd Qwen2-VL-Finetune
bash scripts/finetune_lora.sh