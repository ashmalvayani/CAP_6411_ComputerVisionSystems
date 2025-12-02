import torch
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
from pathlib import Path
from typing import List, Tuple
import os

HF_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
SPLIT = "test"
START_1BASED = 311
END_1BASED = 320
IMAGE_SIZE = 1024
OUTDIR = Path("sdxl_turbo_out")
SEED_BASE = 20251009
SDXL_TURBO_MODEL = "stabilityai/sdxl-turbo"

# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prompt_indices():
    # Convert [311..320] 1-based to zero-based [310..319]
    return list(range(START_1BASED - 1, END_1BASED))
    
def get_prompts() -> List[str]:
    ds = load_dataset(HF_DATASET, split=SPLIT)
    # column could be 'Prompt' or 'prompt' depending on parquet conversion
    col = "Prompt" if "Prompt" in ds.features else ("prompt" if "prompt" in ds.features else None)
    if col is None:
        # fallback to first column
        col = list(ds.features.keys())[0]
    idxs = prompt_indices()
    return [ds[i][col] for i in idxs]

def main():
    print("Loading prompts...")
    prompts = get_prompts()
    
    # Load SDXL-Turbo pipeline (fp16 for speed/memory)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    
    # Generate and save 5 images per prompt
    output_dir = "sdxl_turbo_dataset"
    os.makedirs(output_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts, start=311):     # prompt IDs 311–320
        for img_num in range(1, 100):  # 5 images per prompt
            image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024).images[0]
            img_path = f"{output_dir}/prompt{idx}_img{img_num}.png"
            image.save(img_path)
            print(f"Saved image for prompt {idx}: {img_path}")

if __name__ == "__main__":
    main()
