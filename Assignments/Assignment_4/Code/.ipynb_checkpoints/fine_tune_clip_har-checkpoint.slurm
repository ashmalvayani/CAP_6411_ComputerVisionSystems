#!/bin/bash
#SBATCH --job-name=CLIP
#SBATCH --error=jobs/CLIP-%J.err
#SBATCH --output=jobs/CLIP-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass4

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

python fine_tune_har.py \
    --data_root /home/ashmal/Courses/CVS/Assignment_1/data/HumanActionRecognition/Structured \
    --model_name clip \
    --output_dir ./clip_har_finetuned \
    --num_train_epochs 6 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 5e-5