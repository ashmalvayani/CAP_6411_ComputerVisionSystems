#!/bin/bash
#SBATCH --job-name=WAN-Inference
#SBATCH --error=jobs/WAN-Inference-%J.err
#SBATCH --output=jobs/WAN-Inference-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem48

conda activate /home/ashmal/anaconda3/envs/cvs_ass8

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

cd Wan2.1
python har_wan_generate.py \
  --ckpt_dir /home/ashmal/models/wan21_t2v_13b_diff \
  --size 832*480 \
  --out_root /home/ashmal/Courses/CVS/Assignment_8/Wan2.1/videos/har_videos \
  --num_per_class 10 \
  --num_frames 61 \
  --num_inference_steps 32 \
  --guidance_scale 6.0