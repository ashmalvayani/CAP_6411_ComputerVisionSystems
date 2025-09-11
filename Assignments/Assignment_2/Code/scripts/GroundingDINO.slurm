#!/bin/bash
#SBATCH --job-name=GroundingDINO
#SBATCH --error=outs/GroundingDINO-%J.err
#SBATCH --output=outs/GroundingDINO-%J.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -C gmem24

conda activate /home/ashmal/anaconda3/envs/cvs_ass2

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

cd GroundingDINO

python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
 -p /home/ashmal/Courses/CVS/Assignment_2/scripts/GroundingDINO/weights/groundingdino_swint_ogc.pth \
 --anno_path /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/annotations/instances_val2017.json \
 --image_dir /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/val2017