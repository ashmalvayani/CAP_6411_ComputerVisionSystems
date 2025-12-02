# tdm_sd3_eval.py
# -----------------------------------------------------
# TDM (SD3.5 LoRA) vs plain SD3.5 on Gustavosta prompts 311–320
# - Measures speed per prompt
# - Computes FID vs a COCO subset (same setup as your SDXL script)
# -----------------------------------------------------

import time, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor, Resize, InterpolationMode, Compose
from PIL import Image
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from diffusers import (
    StableDiffusion3Pipeline,
    AutoencoderTiny,
    DPMSolverMultistepScheduler,
)

# ----------------------------
# Config
# ----------------------------
HF_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
SPLIT = "test"
START_1BASED = 311          # inclusive
END_1BASED = 320            # inclusive
IMAGE_SIZE = 1024
OUTDIR = Path("tdm_sd3_eval_out")
SEED_BASE = 20251009

SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
TDM_LORA_REPO = "TDM/TDM-Finetuning-Lora"
SANA_SCHEDULER_REPO = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"

# Student (TDM) vs Teacher (no TDM) settings – taken from the TDM example
TDM_STEPS = 4
TDM_GUIDANCE = 1.0
TEACHER_STEPS = 28
TEACHER_GUIDANCE = 7.0

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
    # column could be 'Prompt' or 'prompt'
    col = "Prompt" if "Prompt" in ds.features else ("prompt" if "prompt" in ds.features else None)
    if col is None:
        col = list(ds.features.keys())[0]
    idxs = prompt_indices()
    return [ds[i][col] for i in idxs]


@dataclass
class GenConfig:
    name: str
    steps: int
    guidance: float
    out_subdir: Path
    use_tdm: bool
    flow_shift: Optional[int] = None  # 6 for TDM, None for teacher


def create_sd3_tdm_pipeline() -> StableDiffusion3Pipeline:
    """
    Create SD3.5 pipeline, load TDM LoRA once, attach Tiny VAE.
    We'll toggle TDM on/off and scheduler settings per run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD3_MODEL,
        torch_dtype=dtype,
    ).to(device)

    # Tiny VAE to save GPU memory (same as example)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=dtype)
    pipe.vae.config.shift_factor = 0.0

    # Load TDM LoRA adapter once
    pipe.load_lora_weights(TDM_LORA_REPO, adapter_name="tdm")

    return pipe


def prepare_for_config(pipe: StableDiffusion3Pipeline, cfg: GenConfig):
    """
    For each configuration (TDM vs teacher), reset scheduler and adapter scales
    following the official TDM example.
    """
    # Always start from the Sana scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        SANA_SCHEDULER_REPO,
        subfolder="scheduler",
    )

    # For TDM: set flow_shift and re-create scheduler
    if cfg.flow_shift is not None:
        pipe.scheduler.config["flow_shift"] = cfg.flow_shift
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Toggle LoRA scale
    if cfg.use_tdm:
        # IMPORTANT: LoRA scale 0.125
        pipe.set_adapters(["tdm"], [0.125])
    else:
        # Disable LoRA (scale 0)
        try:
            pipe.set_adapters(["tdm"], [0.0])
        except Exception:
            # If adapters not loaded correctly, ignore
            pass
    pipe.vae.to(pipe.device)


def derive_seed_for_prompt(prompt_idx_zero_based: int) -> int:
    # Deterministic per-prompt seed for reproducibility
    return (SEED_BASE * 1009 + prompt_idx_zero_based * 9973) % (2**31 - 1)


def generate_and_time(
    pipe: StableDiffusion3Pipeline,
    prompts: List[str],
    cfg: GenConfig,
) -> List[Tuple[str, float, Path]]:
    cfg.out_subdir.mkdir(parents=True, exist_ok=True)
    times: List[Tuple[str, float, Path]] = []

    start_idx0 = START_1BASED - 1

    # Prepare scheduler + LoRA settings for this config
    prepare_for_config(pipe, cfg)

    for i, p in enumerate(tqdm(prompts, desc=f"Generating with {cfg.name}")):
        seed = derive_seed_for_prompt(start_idx0 + i)
        set_seed(seed)
        generator = torch.manual_seed(seed)

        start = time.perf_counter()
        image = pipe(
            prompt=p,
            negative_prompt="",
            num_inference_steps=cfg.steps,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_images_per_prompt=1,
            guidance_scale=cfg.guidance,
            generator=generator,
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
    image_col: Optional[str],
    num: int,
    seed: int,
    outdir: Path,
):
    """
    Samples 'num' images from COCO val mirror. Saves PNGs for reproducibility and
    returns a callable that yields uint8 [B,3,299,299] batches for torchmetrics.
    """
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
        im.save(outdir / f"coco_ref_{i:04d}.png")
        t = transform(im).unsqueeze(0)           # [1,3,299,299] float
        t = (t * 255.0).clamp(0, 255).byte()     # uint8 as torchmetrics expects
        tensors.append(t)

    def _iter(bsz: int = 16):
        for j in range(0, len(tensors), bsz):
            yield torch.cat(tensors[j:j+bsz], dim=0)

    return _iter, split, image_col


def compute_fid_against_reference(ref_iter_fn, gen_folder: Path, device: Optional[str] = None) -> float:
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
    (OUTDIR / "tdm_student").mkdir(exist_ok=True, parents=True)
    (OUTDIR / "teacher").mkdir(exist_ok=True, parents=True)

    print("Loading prompts...")
    prompts = get_prompts()
    with open(OUTDIR / "prompts_311_320.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, start=START_1BASED):
            f.write(f"{i}: {p}\n")

    print("Loading SD3.5 + TDM pipeline (first time can be slow)...")
    pipe = create_sd3_tdm_pipeline()

    cfg_tdm = GenConfig(
        name="SD3.5_TDM_student",
        steps=TDM_STEPS,
        guidance=TDM_GUIDANCE,
        out_subdir=OUTDIR / "tdm_student",
        use_tdm=True,
        flow_shift=6,        # IMPORTANT: for TDM
    )

    cfg_teacher = GenConfig(
        name="SD3.5_teacher_no_TDM",
        steps=TEACHER_STEPS,
        guidance=TEACHER_GUIDANCE,
        out_subdir=OUTDIR / "teacher",
        use_tdm=False,
        flow_shift=None,     # no flow_shift for teacher
    )

    # Generate images and record timings
    tdm_times = generate_and_time(pipe, prompts, cfg_tdm)
    teacher_times = generate_and_time(pipe, prompts, cfg_teacher)

    # --- FID vs COCO val subset ---
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

    print("Computing FID: SD3.5 + TDM (student) vs COCO subset...")
    fid_tdm_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "tdm_student")
    print(f"FID (TDM student vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_tdm_vs_ref:.3f}")

    # Rebuild iterator (we consumed it above)
    ref_iter_fn, _, _ = load_reference_subset_coco(
        dataset=REF_DATASET,
        candidate_splits=REF_CANDIDATE_SPLITS,
        image_col=REF_IMAGE_COLUMN,
        num=REF_NUM_IMAGES,
        seed=REF_SEED,
        outdir=REF_OUTDIR,
    )
    print("Computing FID: SD3.5 teacher (no TDM) vs COCO subset...")
    fid_teacher_vs_ref = compute_fid_against_reference(ref_iter_fn, OUTDIR / "teacher")
    print(f"FID (Teacher vs COCO[{used_split}:{REF_NUM_IMAGES}]): {fid_teacher_vs_ref:.3f}")

    # Save timings CSV (speed comparison)
    rows = []
    for (p, t_student, fp_student), (_, t_teacher, fp_teacher) in zip(tdm_times, teacher_times):
        rows.append({
            "prompt": p,
            "tdm_student_time_sec": t_student,
            "teacher_time_sec": t_teacher,
            "image_tdm_student": str(fp_student),
            "image_teacher": str(fp_teacher),
        })
    df = pd.DataFrame(rows)
    df["speedup_teacher_vs_tdm"] = df["teacher_time_sec"] / df["tdm_student_time_sec"]
    df.to_csv(OUTDIR / "timings_tdm_vs_teacher.csv", index=False)

    print("Done:")
    print(f" - Images: {OUTDIR / 'tdm_student'} and {OUTDIR / 'teacher'}")
    print(f" - Timings CSV: {OUTDIR / 'timings_tdm_vs_teacher.csv'}")
    print(f" - FID(TDM): {fid_tdm_vs_ref:.3f}")
    print(f" - FID(Teacher): {fid_teacher_vs_ref:.3f}")


if __name__ == "__main__":
    main()
