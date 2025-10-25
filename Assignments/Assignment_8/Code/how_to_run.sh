
conda create -n cvs_ass8 python=3.10 -y
cd Assignment_8

git clone https://github.com/Wan-Video/Wan2.1
git clone https://github.com/tdrussell/diffusion-pipe

# download the huggingface model for training
python -m huggingface_hub.download_repo     --repo_id Wan-AI/Wan2.1-T2V-1.3B     --local_dir Wan2.1_model

# download the huggingface model for inference
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ~/models/wan21_t2v_13b_diff

pip install -r requirements.txt

# For training
sbatch run.slurm

# Inference on detailed prompts
sbatch har_wan_generate.slurm

# Inference on simple prompts
sbatch har_wan_generate_simple.slurm

# Single video generation
cd Wan2.1
python wan_generate_like.py \
  --task t2v-1.3B --size 832*480 \
  --prompt "A young boy walking towards the school" \
  --ckpt_dir /home/ashmal/models/wan21_t2v_13b_diff