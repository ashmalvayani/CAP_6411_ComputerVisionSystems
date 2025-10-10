# SDXL vs SDXL-Turbo on Gustavosta test prompts 311–320
# -----------------------------------------------------

import time, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

from torchvision.transforms import ToTensor, Resize, InterpolationMode, Compose

# ----------------------------
# Config
# ----------------------------
HF_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
SPLIT = "test"
START_1BASED = 311
END_1BASED = 320
IMAGE_SIZE = 1024
OUTDIR = Path("sdxl_eval_out")
SEED_BASE = 20251009
SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO_MODEL = "stabilityai/sdxl-turbo"

# Strict per-assignment steps
SDXL_BASE_STEPS = 1000
SDXL_TURBO_STEPS = 4

# Reasonable CFG scales
SDXL_BASE_GUIDANCE = 7.5
SDXL_TURBO_GUIDANCE = 0.0
BATCH_SIZE = 1


# ----- Reference dataset for FID (COCO 2014 val) -----
REF_DATASET = "sayakpaul/coco-30-val-2014"
REF_CANDIDATE_SPLITS = ["validation", "val", "val2014"]
REF_IMAGE_COLUMN = None
REF_NUM_IMAGES = 100
REF_SEED = 1234
REF_OUTDIR = OUTDIR / "ref_coco_subset"
FID_IMAGE_SIZE = 299


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

@dataclass
class GenConfig:
    model_id: str
    steps: int
    guidance: float
    out_subdir: Path

def make_pipeline(model_id: str) -> StableDiffusionXLPipeline:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        dtype=dtype,                 # ← use dtype (not torch_dtype)
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    # Scheduler choice
    if "turbo" in model_id.lower():
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    return pipe

def derive_seed_for_prompt(prompt_idx_zero_based: int) -> int:
    # Deterministic per-prompt seed
    return (SEED_BASE * 1009 + prompt_idx_zero_based * 9973) % (2**31 - 1)

def generate_and_time(pipe: StableDiffusionXLPipeline, prompts: List[str], cfg: GenConfig) -> List[Tuple[str, float, Path]]:
    cfg.out_subdir.mkdir(parents=True, exist_ok=True)
    times = []
    start_idx0 = START_1BASED - 1
    for i, p in enumerate(tqdm(prompts, desc=f"Generating with {cfg.model_id}")):
        seed = derive_seed_for_prompt(start_idx0 + i)
        set_seed(seed)
        start = time.perf_counter()
        image = pipe(
            p,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]
        elapsed = time.perf_counter() - start
        fn = cfg.out_subdir / f"{START_1BASED + i:04d}.png"
        image.save(fn)
        times.append((p, elapsed, fn))
    return times


def _pick_available_split(dataset_name: str, candidates: list[str]) -> str:
    # For sayakpaul/coco-30-val-2014, everything is under 'train'
    dset_dict = load_dataset(dataset_name)
    for s in candidates:
        if s in dset_dict:
            return s
    # fallback to first available
    return list(dset_dict.keys())[0]

def load_reference_subset_coco(
    dataset: str,
    candidate_splits: list[str],
    image_col: str | None,
    num: int,
    seed: int,
    outdir: Path,
):
    """
    Samples 'num' images from COCO val mirror. Saves PNGs for reproducibility and
    returns a callable that yields uint8 [B,3,299,299] batches for torchmetrics.
    """
    import random

    split = _pick_available_split(dataset, candidate_splits)
    ds = load_dataset(dataset, split=split)

    # Auto-detect image column
    if image_col is None:
        for cand in ("image", "img", "Image", "images"):
            if cand in ds.features:
                image_col = cand
                break
        if image_col is None:
            for k, v in ds.features.items():
                if "Image(" in str(v):
                    image_col = k
                    break
    if image_col is None:
        raise ValueError("Could not find an image column in the reference dataset.")

    idxs = list(range(len(ds)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    idxs = idxs[:num]

    outdir.mkdir(parents=True, exist_ok=True)

    transform = Compose([
        Resize((FID_IMAGE_SIZE, FID_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),  # [0,1] float
    ])

    tensors = []
    for i, idx in enumerate(idxs):
        im = ds[idx][image_col].convert("RGB")
        # save a viewable PNG copy (optional)
        im.save(outdir / f"coco_ref_{i:04d}.png")
        t = transform(im).unsqueeze(0)           # [1,3,299,299] float
        t = (t * 255.0).clamp(0, 255).byte()     # uint8 as torchmetrics expects
        tensors.append(t)

    def _iter(bsz: int = 16):
        for j in range(0, len(tensors), bsz):
            yield torch.cat(tensors[j:j+bsz], dim=0)

    return _iter, split, image_col

def compute_fid_against_reference(ref_iter_fn, gen_folder: Path, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # REAL (reference)
    for xb in ref_iter_fn():
        fid.update(xb.to(device), real=True)

    # FAKE (generated)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    resize_to_299 = Compose([
        Resize((FID_IMAGE_SIZE, FID_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
    ])

    batch = []
    for p in sorted(gen_folder.iterdir()):
        if p.suffix.lower() in exts:
            img = Image.open(p).convert("RGB")
            x = resize_to_299(img).unsqueeze(0)         # [1,3,299,299] float
            x = (x * 255.0).clamp(0, 255).byte()
            batch.append(x)
            if len(batch) == 16:
                fid.update(torch.cat(batch, dim=0).to(device), real=False)
                batch = []
    if batch:
        fid.update(torch.cat(batch, dim=0).to(device), real=False)

    return float(fid.compute().item())


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "sdxl_base").mkdir(exist_ok=True, parents=True)
    (OUTDIR / "sdxl_turbo").mkdir(exist_ok=True, parents=True)

    print("Loading prompts...")
    prompts = get_prompts()
    with open(OUTDIR / "prompts_311_320.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, start=START_1BASED):
            f.write(f"{i}: {p}\n")

    print("Loading pipelines (first time can be slow)...")
    pipe_base = make_pipeline(SDXL_BASE_MODEL)
    pipe_turbo = make_pipeline(SDXL_TURBO_MODEL)

    cfg_base = GenConfig(SDXL_BASE_MODEL, SDXL_BASE_STEPS, SDXL_BASE_GUIDANCE, OUTDIR / "sdxl_base")
    cfg_turbo = GenConfig(SDXL_TURBO_MODEL, SDXL_TURBO_STEPS, SDXL_TURBO_GUIDANCE, OUTDIR / "sdxl_turbo")

    base_times = generate_and_time(pipe_base, prompts, cfg_base)
    turbo_times = generate_and_time(pipe_turbo, prompts, cfg_turbo)

    # --- NEW: FID vs COCO val subset (100 images) ---
    print("Preparing COCO val2014 reference subset for FID...")
    ref_iter_fn, used_split, used_img_col = load_reference_subset_coco(
        dataset=REF_DATASET,
        candidate_splits=REF_CANDIDATE_SPLITS,
        image_col=REF_IMAGE_COLUMN,
        num=REF_NUM_IMAGES,
        seed=REF_SEED,
        outdir=REF_OUTDIR,
    )
    print(f"Reference: {REF_DATASET} / split='{used_split}', column='{used_img_col}', N={REF_NUM_IMAGES}")

    print("Computing FID: SDXL-Base vs COCO subset...")
    fid_base_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "sdxl_base")
    print(f"FID (SDXL-Base vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_base_vs_ref:.3f}")

    # Rebuild iterator (we consumed it above)
    ref_iter_fn, _, _ = load_reference_subset_coco(
        dataset=REF_DATASET,
        candidate_splits=REF_CANDIDATE_SPLITS,
        image_col=REF_IMAGE_COLUMN,
        num=REF_NUM_IMAGES,
        seed=REF_SEED,
        outdir=REF_OUTDIR,
    )
    print("Computing FID: SDXL-Turbo vs COCO subset...")
    fid_turbo_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "sdxl_turbo")
    print(f"FID (SDXL-Turbo vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_turbo_vs_ref:.3f}")

    rows = []
    for (p, t, fp), (_, t2, fp2) in zip(base_times, turbo_times):
        rows.append({
            "prompt": p,
            "sdxl_base_time_sec": t,
            "sdxl_turbo_time_sec": t2,
            "image_base": str(fp),
            "image_turbo": str(fp2),
        })
    df = pd.DataFrame(rows)
    df["speedup_turbo_vs_base"] = df["sdxl_base_time_sec"] / df["sdxl_turbo_time_sec"]
    df.to_csv(OUTDIR / "timings.csv", index=False)

    print("Inference Done:")
    print(f" - Images: {OUTDIR / 'sdxl_base'} and {OUTDIR / 'sdxl_turbo'}")
    print(f" - CSV:    {OUTDIR / 'timings.csv'}")

if __name__ == "__main__":
    main()
