"""
evaluate_fvd_scores.py
----------------------------------------------------------
Alternative implementation of FVD evaluation for two models:
(1) Wan 2.1 (Finetuned) and (2) FramePack (Pretrained).
Loads videos, resamples frames, and computes Fréchet Video Distance (FVD)
using the official `common_metrics_on_video_quality` repository.
----------------------------------------------------------
"""

import torch
from torchvision import transforms
from decord import VideoReader, cpu
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys

# ------------------------------------------------------------
# Import FVD metric
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
METRIC_REPO = ROOT_DIR / "common_metrics_on_video_quality"
if str(METRIC_REPO) not in sys.path:
    sys.path.append(str(METRIC_REPO))

try:
    from calculate_fvd import calculate_fvd
except ImportError as e:
    raise ImportError(
        f"Cannot import 'calculate_fvd'. Ensure repo is cloned:\n"
        f"git clone https://github.com/JunyaoHu/common_metrics_on_video_quality.git"
    ) from e


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODELS = {
    "Wan2.1 (Finetuned)": Path("./generated_videos_wan"),
    "FramePack (Pretrained)": Path("./generated_videos_framepack")
}

REAL_VIDEOS = '/home/ashmal/Courses/CVS/Assignment_8/data/HAR-Dataset/HumanActivityRecognition-Dataset'

RESOLUTION = 224
NUM_FRAMES = 16


# ------------------------------------------------------------
# Video loading
# ------------------------------------------------------------
def load_video_folder(folder: Path, num_frames: int, res: int) -> torch.Tensor:
    """Load all videos from a folder (recursively) into a tensor of shape (N, T, C, H, W)."""
    video_paths = list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi"))
    if not video_paths:
        print(f"[WARN] No videos found in: {folder}")
        return torch.empty(0)

    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor()
    ])

    tensors = []
    print(f"Loading {len(video_paths)} videos from {folder.name} ...")
    for path in tqdm(video_paths, desc=f"Reading {folder.name}"):
        try:
            reader = VideoReader(str(path), ctx=cpu(0))
            total = len(reader)
            idx = torch.linspace(0, total - 1, num_frames).long()
            frames = reader.get_batch(idx).asnumpy()
            clip = torch.stack([transform(Image.fromarray(f)) for f in frames], dim=0)
            tensors.append(clip)
        except Exception as e:
            print(f"  [SKIP] {path.name} → {e}")

    return torch.stack(tensors) if tensors else torch.empty(0)


# ------------------------------------------------------------
# FVD computation
# ------------------------------------------------------------
def compute_fvd(real_tensor: torch.Tensor, gen_tensor: torch.Tensor, device: str) -> float:
    """Compute FVD given real and generated tensors."""
    num = min(len(real_tensor), len(gen_tensor))
    if num == 0:
        return float("nan")
    subset_real = real_tensor[:num].to(device)
    subset_fake = gen_tensor[:num].to(device)

    result = calculate_fvd(
        subset_real, subset_fake,
        device=device,
        method="styleganv",
        only_final=True
    )
    return result["value"][0]


# ------------------------------------------------------------
# Evaluation runner
# ------------------------------------------------------------
def run_fvd():
    print("=" * 25, "FVD EVALUATION", "=" * 25)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load reference videos
    if not REAL_VIDEOS.exists():
        print(f"[ERROR] Dataset not found at {REAL_VIDEOS}")
        return
    real_tensor = load_video_folder(REAL_VIDEOS, NUM_FRAMES, RESOLUTION)
    if real_tensor.numel() == 0:
        print("[ERROR] No real videos could be loaded — aborting.")
        return

    print(f"Loaded {len(real_tensor)} real videos.\n")

    scores = {}
    for name, model_dir in MODELS.items():
        print(f"--- Evaluating {name} ---")
        if not model_dir.exists():
            print(f"[WARN] Missing directory: {model_dir}")
            continue

        gen_tensor = load_video_folder(model_dir, NUM_FRAMES, RESOLUTION)
        if gen_tensor.numel() == 0:
            print(f"[WARN] No generated videos found for {name}.")
            continue

        print(f"Comparing {min(len(real_tensor), len(gen_tensor))} videos...")
        try:
            score = compute_fvd(real_tensor, gen_tensor, device)
            scores[name] = score
            print(f"FVD for {name}: {score:.4f}\n")
        except Exception as e:
            print(f"[ERROR] FVD failed for {name}: {e}\n")

    print("=" * 25, "FINAL RESULTS", "=" * 25)
    if not scores:
        print("No FVD results available.")
    else:
        for k, v in scores.items():
            print(f"{k:<25}: {v:.4f}")
    print("=" * 63)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_fvd()
