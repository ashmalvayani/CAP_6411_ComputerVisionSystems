#!/bin/bash
#SBATCH --job-name=SophiaVL
#SBATCH --error=jobs/infer_SophiaVL-%J.err
#SBATCH --output=jobs/infer_SophiaVL-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass5_qwen25vl

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

python infer_mathvista_sophiavl.py \
  --model_id bunny127/SophiaVL-R1-7B \
  --lora_path Qwen2-VL-Finetune/output/testing_lora \
  --split test \
  --output results/sophiavl_r1_test.json \
  --dtype bfloat16 --batch_size 1 --temperature 0.0