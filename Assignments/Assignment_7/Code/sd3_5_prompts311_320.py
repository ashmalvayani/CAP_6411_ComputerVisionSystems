# SD3.5 Large vs SD3.5 Large-Turbo on Gustavosta test prompts 311–320
# ------------------------------------------------------------------

import os, time, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Compose
from PIL import Image
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

# ----------------------------
# Config
# ----------------------------
HF_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
SPLIT = "test"
START_1BASED = 311
END_1BASED = 320
IMAGE_SIZE = 1024
OUTDIR = Path("sd35_eval_out")
SEED_BASE = 20251009

SD35_LARGE_MODEL = "stabilityai/stable-diffusion-3.5-large"
SD35_TURBO_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"

# Per-assignment steps (sane SD3 defaults)
SD35_LARGE_STEPS = 28
SD35_TURBO_STEPS = 4

# CFG scales (SD3 docs: 5–7 typical; Turbo lower CFG works well)
SD35_LARGE_GUIDANCE = 7.0
SD35_TURBO_GUIDANCE = 3.0

BATCH_SIZE = 1  # keep as in your original

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
    return list(range(START_1BASED - 1, END_1BASED))

def get_prompts() -> List[str]:
    ds = load_dataset(HF_DATASET, split=SPLIT)
    col = "Prompt" if "Prompt" in ds.features else ("prompt" if "prompt" in ds.features else None)
    if col is None:
        col = list(ds.features.keys())[0]
    idxs = prompt_indices()
    return [ds[i][col] for i in idxs]

@dataclass
class GenConfig:
    model_id: str
    steps: int
    guidance: float
    out_subdir: Path

def _maybe_hf_login():
    """
    If HUGGINGFACE_TOKEN is present, use it to avoid interactive login
    for gated SD3.5 models.
    """
    tok = os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        try:
            from huggingface_hub import login
            login(tok, add_to_git_credential=False)
        except Exception:
            pass

def make_pipeline(model_id: str) -> StableDiffusion3Pipeline:
    _maybe_hf_login()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,          # SD3 docs recommend fp16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If VRAM is tight, flip this on; it trades a little latency for memory.
    # pipe.enable_model_cpu_offload()

    pipe = pipe.to(device)
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    return pipe

def derive_seed_for_prompt(prompt_idx_zero_based: int) -> int:
    return (SEED_BASE * 1009 + prompt_idx_zero_based * 9973) % (2**31 - 1)

def generate_and_time(pipe: StableDiffusion3Pipeline, prompts: List[str], cfg: GenConfig) -> List[Tuple[str, float, Path]]:
    cfg.out_subdir.mkdir(parents=True, exist_ok=True)
    times = []
    start_idx0 = START_1BASED - 1
    for i, p in enumerate(tqdm(prompts, desc=f"Generating with {cfg.model_id}")):
        seed = derive_seed_for_prompt(start_idx0 + i)
        set_seed(seed)
        start = time.perf_counter()
        image = pipe(
            prompt=p,
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
    dset_dict = load_dataset(dataset_name)
    for s in candidates:
        if s in dset_dict:
            return s
    return list(dset_dict.keys())[0]

def load_reference_subset_coco(
    dataset: str,
    candidate_splits: list[str],
    image_col: str | None,
    num: int,
    seed: int,
    outdir: Path,
):
    import random
    split = _pick_available_split(dataset, candidate_splits)
    ds = load_dataset(dataset, split=split)

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
        ToTensor(),
    ])

    tensors = []
    for i, idx in enumerate(idxs):
        im = ds[idx][image_col].convert("RGB")
        im.save(outdir / f"coco_ref_{i:04d}.png")
        t = transform(im).unsqueeze(0)
        t = (t * 255.0).clamp(0, 255).byte()
        tensors.append(t)

    def _iter(bsz: int = 16):
        for j in range(0, len(tensors), bsz):
            yield torch.cat(tensors[j:j+bsz], dim=0)

    return _iter, split, image_col

def compute_fid_against_reference(ref_iter_fn, gen_folder: Path, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # REAL
    for xb in ref_iter_fn():
        fid.update(xb.to(device), real=True)

    # FAKE
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    resize_to_299 = Compose([
        Resize((FID_IMAGE_SIZE, FID_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
    ])

    batch = []
    for p in sorted(gen_folder.iterdir()):
        if p.suffix.lower() in exts:
            img = Image.open(p).convert("RGB")
            x = resize_to_299(img).unsqueeze(0)
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
    (OUTDIR / "sd35_large").mkdir(exist_ok=True, parents=True)
    (OUTDIR / "sd35_large_turbo").mkdir(exist_ok=True, parents=True)

    print("Loading prompts...")
    prompts = get_prompts()
    with open(OUTDIR / "prompts_311_320.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, start=START_1BASED):
            f.write(f"{i}: {p}\n")

    print("Loading pipelines (first time can be slow)...")
    pipe_large = make_pipeline(SD35_LARGE_MODEL)
    pipe_turbo = make_pipeline(SD35_TURBO_MODEL)

    cfg_large = GenConfig(SD35_LARGE_MODEL, SD35_LARGE_STEPS, SD35_LARGE_GUIDANCE, OUTDIR / "sd35_large")
    cfg_turbo = GenConfig(SD35_TURBO_MODEL, SD35_TURBO_STEPS, SD35_TURBO_GUIDANCE, OUTDIR / "sd35_large_turbo")

    large_times = generate_and_time(pipe_large, prompts, cfg_large)
    turbo_times = generate_and_time(pipe_turbo, prompts, cfg_turbo)

    # --- FID vs COCO val subset (100 images) ---
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

    print("Computing FID: SD3.5 Large vs COCO subset...")
    fid_large_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "sd35_large")
    print(f"FID (SD3.5 Large vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_large_vs_ref:.3f}")

    ref_iter_fn, _, _ = load_reference_subset_coco(
        dataset=REF_DATASET,
        candidate_splits=REF_CANDIDATE_SPLITS,
        image_col=REF_IMAGE_COLUMN,
        num=REF_NUM_IMAGES,
        seed=REF_SEED,
        outdir=REF_OUTDIR,
    )
    print("Computing FID: SD3.5 Large-Turbo vs COCO subset...")
    fid_turbo_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "sd35_large_turbo")
    print(f"FID (SD3.5 Large-Turbo vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_turbo_vs_ref:.3f}")

    rows = []
    for (p, t, fp), (_, t2, fp2) in zip(large_times, turbo_times):
        rows.append({
            "prompt": p,
            "sd35_large_time_sec": t,
            "sd35_large_turbo_time_sec": t2,
            "image_large": str(fp),
            "image_large_turbo": str(fp2),
        })
    df = pd.DataFrame(rows)
    df["speedup_turbo_vs_large"] = df["sd35_large_time_sec"] / df["sd35_large_turbo_time_sec"]
    df.to_csv(OUTDIR / "timings.csv", index=False)

    print("Inference Done:")
    print(f" - Images: {OUTDIR / 'sd35_large'} and {OUTDIR / 'sd35_large_turbo'}")
    print(f" - CSV:    {OUTDIR / 'timings.csv'}")

if __name__ == "__main__":
    main()
