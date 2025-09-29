#!/bin/bash
#SBATCH --job-name=InternVL3
#SBATCH --error=jobs/infer_InternVL3-%J.err
#SBATCH --output=jobs/infer_InternVL3-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass5_internvl

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

python infer_mathvista_internvl3.py \
  --model_id OpenGVLab/InternVL3-8B \
  --lora_path InternVL/internvl_chat/work_dirs/internvl_chat_v3/internvl3_8b_dynamic_res_2nd_finetune_full \
  --split test \
  --output results/internvl3_test.json \
  --dtype bfloat16 --batch_size 1 --temperature 0.0