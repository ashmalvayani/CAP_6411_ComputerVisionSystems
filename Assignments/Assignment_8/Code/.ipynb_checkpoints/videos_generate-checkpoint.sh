# Example paths (edit to yours)
BASE=/home/ashmal/Courses/CVS/Assignment_8/diffusion-pipe/Wan2.1-T2V-1.3B
# BASE="Wan-AI/Wan2.1-T2V-1.3B"
LORA=/home/ashmal/Courses/CVS/Assignment_8/diffusion-pipe/finetuned_outputs/wan_lora_output/20251024_08-06-40/epoch10
OUT=/home/ashmal/Courses/CVS/Assignment_8/diffusion-pipe/finetuned_outputs/har_generations

python generate_wan_lora_videos.py \
  --base_model "$BASE" \
  --lora_dir "$LORA" \
  --out_dir "$OUT" \
  --per_class 10 \
  --width 832 --height 480 \
  --num_frames 65 --fps 8 \
  --steps 30 --guidance 3.5 \
  --seed 1234
