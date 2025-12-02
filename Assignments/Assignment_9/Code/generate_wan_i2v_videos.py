"""
python generate_wan_i2v_videos.py \
  --base_model Wan-Video/Wan2.1-I2V-1.3B \
  --images_root ./Assignment_9/Images \
  --out_root ./Assignment_9/generated_videos_wan \
  --fp16
"""

import os, time, argparse, torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from transformers import AutoTokenizer
from PIL import Image

# ------------------------------------------------------------
# Classes and prompt components
# ------------------------------------------------------------
CLASSES = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone",
]

# short single-person subject variants (for variety)
SUBJECTS = [
    "A man", "A woman", "A boy", "A girl", "A child",
    "An elderly person", "A student", "An office worker",
    "A person", "A teenager",
]

# ------------------------------------------------------------
# Build the Wan 2.1 I2V pipeline
# ------------------------------------------------------------
def build_pipeline(base_model, fp16=False):
    dtype = torch.float16 if fp16 else torch.bfloat16
    tok = AutoTokenizer.from_pretrained("google/umt5-xxl", use_fast=True)
    pipe = DiffusionPipeline.from_pretrained(
        base_model,                      # e.g. "Wan-Video/Wan2.1-I2V-1.3B"
        tokenizer=tok,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    pipe.to("cuda")
    try:
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


# ------------------------------------------------------------
# Video generation per image
# ------------------------------------------------------------
def generate_one(pipe, image_path, prompt, seed, w, h, num_frames, steps, guidance, fps, out_path):
    g = torch.Generator(device="cuda").manual_seed(seed)
    image = Image.open(image_path).convert("RGB").resize((w, h))
    result = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_frames=num_frames,
        generator=g,
    )
    frames = result.frames[0]
    export_to_video(frames, out_path, fps=fps)


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="e.g. Wan-Video/Wan2.1-I2V-1.3B")
    ap.add_argument("--images_root", default="./Assignment_9/Images")
    ap.add_argument("--out_root", default="./Assignment_9/Outputs")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    pipe = build_pipeline(args.base_model, args.fp16)

    # loop over each HAR class folder
    for cls_name in sorted(os.listdir(args.images_root)):
        class_dir = os.path.join(args.images_root, cls_name)
        if not os.path.isdir(class_dir):
            continue

        # simple natural-language prompt derived from folder name
        prompt = f"A {torch.choice(torch.tensor(range(len(SUBJECTS)))).item() if False else 'person'} performing the action: {cls_name.lower()}."

        # or if you prefer more variety:
        # prompt = f"{random.choice(SUBJECTS)} performing {cls_name.lower()}."

        print(f"\n=== Generating videos for class: {cls_name} ===")
        print(f"Prompt: {prompt}")

        out_class_dir = os.path.join(args.out_root, cls_name)
        os.makedirs(out_class_dir, exist_ok=True)

        for img_name in sorted(os.listdir(class_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            in_path = os.path.join(class_dir, img_name)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_class_dir, f"{os.path.splitext(img_name)[0]}_{ts}.mp4")

            print(f"→ {cls_name}: {img_name} → {out_path}")
            generate_one(
                pipe, in_path, prompt,
                args.seed, args.width, args.height,
                args.num_frames, args.steps, args.guidance, args.fps, out_path
            )

    print("\n✅ All videos generated and saved under:", args.out_root)


if __name__ == "__main__":
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    main()
