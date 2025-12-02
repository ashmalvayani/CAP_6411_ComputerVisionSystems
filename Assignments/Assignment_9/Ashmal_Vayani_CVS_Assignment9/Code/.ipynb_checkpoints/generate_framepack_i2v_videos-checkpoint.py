"""
generate_framepack_i2v_videos.py
"""

import os
import torch
from diffusers import (
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BASE_DIR = "./Assignment_9/Images" # this path contains images
OUTPUT_DIR = "./Assignment_9/generated_videos_framepack"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Load FramePack I2V components
# -------------------------------------------------------------------------
print("Loading FramePack I2V model...")

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
)

pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)

# Optimize memory for A100 / limited VRAM devices
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# -------------------------------------------------------------------------
# Iterate over each activity folder and generate videos
# -------------------------------------------------------------------------
for activity in sorted(os.listdir(BASE_DIR)):
    class_dir = os.path.join(BASE_DIR, activity)
    if not os.path.isdir(class_dir):
        continue

    # Use the folder name as the natural-language prompt
    prompt = f"A person performing the action: {activity.lower()}."
    print(f"\n=== Generating videos for: {activity} ===")
    print(f"Prompt: {prompt}")

    # Create output directory for this class
    out_class_dir = os.path.join(OUTPUT_DIR, activity)
    os.makedirs(out_class_dir, exist_ok=True)

    # Loop through all images inside this class folder
    for img_name in sorted(os.listdir(class_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_dir, img_name)
        first_image = load_image(img_path)

        # Run FramePack inference
        print(f"→ Processing: {img_name}")
        result = pipe(
            image=first_image,
            prompt=prompt,
            height=512,
            width=512,
            num_frames=25,
            num_inference_steps=51,
            guidance_scale=6.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            sampling_type="inverted_anti_drifting",
        ).frames[0]

        # Export to MP4 (same name as input image)
        video_out_path = os.path.join(out_class_dir, f"{os.path.splitext(img_name)[0]}.mp4")
        export_to_video(result, video_out_path, fps=24)
        print(f"Saved: {video_out_path}")

print("\nAll videos generated successfully and saved in:", OUTPUT_DIR)
