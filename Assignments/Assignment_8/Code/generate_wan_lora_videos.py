# generate_wan_lora_videos.py
import os, time, argparse, torch, random
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from transformers import AutoTokenizer

CATEGORIES = {
    "Clapping": "A person clapping hands, centered, full-body, plain background, 8s video.",
    "Meet and Split": "Two people meet, talk briefly, then split and walk away, plain background, 8s video.",
    "Sitting": "A person sits down on a chair and remains seated, plain background, 8s video.",
    "Standing Still": "A person stands still facing camera, slight natural motion, 8s video.",
    "Walking": "A person walking forward, full-body, plain background, 8s video.",
    "Walking While Reading Book": "A person walking while reading a book, careful steps, 8s video.",
    "Walking While Using Phone": "A person walking while using a phone, texting while walking, 8s video.",
}

def build_pipeline(base_model, lora_dir, fp16=False):
    dtype = torch.float16 if fp16 else torch.bfloat16
    tok = AutoTokenizer.from_pretrained("google/umt5-xxl", use_fast=True)
    
    pipe = DiffusionPipeline.from_pretrained(
        base_model,                # e.g. "Wan-Video/Wan2.1-T2V-1.3B" or a local clone with model_index.json
        tokenizer=tok,
        low_cpu_mem_usage=True,
        # ignore_mismatched_sizes=True,
        torch_dtype=dtype,
        trust_remote_code=True     # WAN uses custom modules
    )
    pipe.load_lora_weights(lora_dir, weight_name="adapter_model.safetensors", adapter_name="har")
    try:
        pipe.fuse_lora()
    except Exception:
        pass
    pipe.to("cuda")
    try:
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe

def generate_one(pipe, prompt, seed, w, h, num_frames, steps, guidance, fps, out_path):
    g = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_frames=num_frames,
        height=h,
        width=w,
        generator=g,
    )
    frames = result.frames[0]
    export_to_video(frames, out_path, fps=fps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)  # "Wan-Video/Wan2.1-T2V-1.3B" or local diffusers folder
    ap.add_argument("--lora_dir", required=True)    # folder with adapter_model.safetensors + adapter_config.json
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_class", type=int, default=10)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=65)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pipe = build_pipeline(args.base_model, args.lora_dir, args.fp16)

    for category, base_prompt in CATEGORIES.items():
        cat_dir = os.path.join(args.out_dir, category.replace(" ", "_"))
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n==> Generating {args.per_class} videos for '{category}'")
        for i in range(args.per_class):
            seed_i = args.seed + i + (abs(hash(category)) % 100000)
            prompt_i = f"{base_prompt} cinematic, neutral lighting, natural motion."
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(cat_dir, f"{category.replace(' ','_')}_{i:02d}_s{seed_i}_{ts}.mp4")
            print(f"[{category}] {i+1}/{args.per_class} -> {out_path}")
            generate_one(pipe, prompt_i, seed_i, args.width, args.height,
                         args.num_frames, args.steps, args.guidance, args.fps, out_path)

if __name__ == "__main__":
    # Optional: ensure flash-attn is not forced
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    main()
