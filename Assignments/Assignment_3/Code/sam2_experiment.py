#!/usr/bin/env python3
"""
CamVid Segmentation with SAM2, CLIPSeg, and YOLOv8
Baselines:
  1. GroundingDINO + SAM2 (text prompt → boxes → masks)
  2. CLIPSeg (direct text-to-segmentation, resized to GT)
Proposed:
  YOLOv8 + SAM2 (detector boxes → masks)

Evaluates Dice scores for Person & Vehicle.
"""

import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt

# ========= UTILS =========
    
def visualize_and_save(image, gt_person, gt_vehicle, pred_person, pred_vehicle, method_name, img_name, out_dir):
    """Overlay predictions vs GT and save visualization"""
    os.makedirs(out_dir, exist_ok=True)

    # GT overlay (red=person, blue=vehicle)
    gt_overlay = image.copy()
    gt_overlay[gt_person == 1] = [255, 0, 0]   # red
    gt_overlay[gt_vehicle == 1] = [0, 0, 255] # blue

    # Prediction overlay (green=person, yellow=vehicle)
    pred_overlay = image.copy()
    pred_overlay[pred_person == 1] = [0, 255, 0]     # green
    pred_overlay[pred_vehicle == 1] = [255, 255, 0]  # yellow

    # Save side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT (Person=Red, Vehicle=Blue)")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"{method_name} Prediction\n(Person=Green, Vehicle=Yellow)")
    axes[2].axis("off")

    save_path = os.path.join(out_dir, f"{img_name}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
def dice_score(pred_mask, gt_mask):
    """Compute Dice score between binary masks"""
    pred = (pred_mask > 0).astype(bool)
    gt   = (gt_mask > 0).astype(bool)
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum())


def load_gt_masks(mask_path):
    """Convert CamVid RGB mask into person & vehicle binary masks"""
    mask = cv2.imread(str(mask_path))[:, :, ::-1]  # BGR→RGB

    # Person classes
    person_colors = [
        (64, 64, 0),      # Pedestrian
        (192, 128, 64),   # Child
        (0, 128, 192),    # Bicyclist
        (192, 0, 192)     # Motorcyclist
    ]

    # Vehicle classes
    vehicle_colors = [
        (64, 0, 128),     # Car
        (64, 128, 192),   # SUVPickupTruck
        (192, 128, 192),  # Truck_Bus
        (64, 0, 192)      # Cart/Luggage/Pram
    ]

    person_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    vehicle_mask = np.zeros_like(person_mask)

    for color in person_colors:
        match = np.all(mask == color, axis=-1)
        person_mask[match] = 1

    for color in vehicle_colors:
        match = np.all(mask == color, axis=-1)
        vehicle_mask[match] = 1

    return person_mask, vehicle_mask


# ========= SETUP MODELS =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM2
sam_checkpoint = "sam_vit_h_4b8939.pth"
if not Path(sam_checkpoint).exists():
    os.system(f"wget -O {sam_checkpoint} https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}")
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(DEVICE)
sam_predictor = SamPredictor(sam)

# YOLOv8
yolo_model = YOLO("yolov8m.pt")

# GroundingDINO
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_ckpt = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
dino_model = load_model(config_path, grounding_dino_ckpt)

# CLIPSeg
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)

# ========= PIPELINES =========

def baseline_dino_sam2(img_path):
    """GroundingDINO + SAM2 with correct box scaling"""
    # Load image for DINO
    image_source, image_tensor = load_image(img_path)
    H, W, _ = cv2.imread(img_path).shape

    # Run DINO
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption="person . bicycle . motorcycle . car . truck . bus",
        box_threshold=0.25,
        text_threshold=0.35
    )

    # Convert boxes from cxcywh (normalized) → xyxy (pixels)
    boxes_xyxy = []
    for b in boxes:
        cx, cy, w, h = b.tolist()
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_xyxy = torch.tensor(boxes_xyxy, device=DEVICE)

    # Load image for SAM
    image = cv2.imread(img_path)[:, :, ::-1]
    sam_predictor.set_image(image)

    # Prepare masks
    person_mask = np.zeros((H, W), dtype=np.uint8)
    vehicle_mask = np.zeros((H, W), dtype=np.uint8)

    # Run SAM for each detected box
    for box_xyxy, phrase in zip(boxes_xyxy, phrases):
        box_t = sam_predictor.transform.apply_boxes_torch(
            box_xyxy.unsqueeze(0), (H, W)
        )
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=box_t,
            multimask_output=False
        )
        mask = (masks[0][0].cpu().numpy() > 0.5).astype(np.uint8)
    
        phrase = phrase.lower()
        if any(v in phrase for v in ["person", "bicycle", "motorcycle"]):
            person_mask |= mask
        elif any(v in phrase for v in ["car", "truck", "bus"]):
            vehicle_mask |= mask

    return person_mask, vehicle_mask

def baseline_clipseg(image, target_shape):
    image_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    person_prompts = ["person", "man", "woman", "bicycle", "motorbike"]
    vehicle_prompts = ["car", "truck", "bus"]

    def run_clipseg(prompts, category_name):
        mask_accum = np.zeros(target_shape, dtype=np.uint8)
        for text in prompts:
            inputs = clipseg_processor(
                text=[text],
                images=[image_pil],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = clipseg_model(**inputs)

            pred = outputs.logits[0]

            mask = torch.sigmoid(pred).cpu().numpy()
            mask_resized = cv2.resize(
                mask, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            mask_bin = (mask_resized > 0.3).astype(np.uint8)
            mask_accum |= mask_bin

        return mask_accum

    person_mask = run_clipseg(person_prompts, "person")
    vehicle_mask = run_clipseg(vehicle_prompts, "vehicle")

    return person_mask, vehicle_mask

def proposed_yolo_sam2(image):
    """YOLOv8 + SAM2 auto detection"""
    results = yolo_model.predict(image, conf=0.25, verbose=False)[0]
    sam_predictor.set_image(image)

    person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    vehicle_mask = np.zeros_like(person_mask)

    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        label = yolo_model.names[int(cls)].lower()
        if label not in ["person", "car", "bus", "truck", "motorbike", "bicycle"]:
            continue

        box_t = sam_predictor.transform.apply_boxes_torch(
            torch.tensor([box], device=DEVICE), image.shape[:2]
        )
        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=box_t, multimask_output=False
        )

        mask = (masks[0][0].cpu().numpy() > 0.5).astype(np.uint8)

        if label in ["person", "bicycle", "motorbike"]:
            person_mask |= mask
        elif label in ["car", "bus", "truck"]:
            vehicle_mask |= mask

    return person_mask, vehicle_mask


# ========= MAIN EXPERIMENT =========

def main():
    camvid_dir = "/home/ashmal/Courses/CVS/Assignment_3/data/CamVid_github"
    dice_dino_person, dice_dino_vehicle = [], []
    dice_clip_person, dice_clip_vehicle = [], []
    dice_prop_person, dice_prop_vehicle = [], []

    img_dir = Path(camvid_dir) / "val"
    mask_dir = Path(camvid_dir) / "val_labels"

    # For visualization control
    vis_count = {"DINO": 0, "CLIPSeg": 0, "YOLO": 0}
    vis_limit = 8

    # >>> TIMING & MEMORY TRACKERS
    stats = {
        "DINO": {"time": 0.0, "samples": 0, "mem": []},
        "CLIPSeg": {"time": 0.0, "samples": 0, "mem": []},
        "YOLO": {"time": 0.0, "samples": 0, "mem": []},
    }

    for img_path in tqdm(sorted(img_dir.glob("*.png"))):
        mask_path = mask_dir / f"{img_path.stem}_L.png"
        gt_person, gt_vehicle = load_gt_masks(mask_path)
        image = cv2.imread(str(img_path))[:, :, ::-1]
        img_name = img_path.stem

        # Baseline 1: GroundingDINO + SAM2
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        pred_p_b, pred_v_b = baseline_dino_sam2(str(img_path))
        elapsed = time.time() - start
        mem_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        stats["DINO"]["time"] += elapsed
        stats["DINO"]["samples"] += 1
        stats["DINO"]["mem"].append(mem_used)

        dice_dino_person.append(dice_score(pred_p_b, gt_person))
        dice_dino_vehicle.append(dice_score(pred_v_b, gt_vehicle))
        if vis_count["DINO"] < vis_limit:
            visualize_and_save(image, gt_person, gt_vehicle,
                               pred_p_b, pred_v_b,
                               "DINO+SAM2", img_name,
                               "Outputs/Baseline_DINO_SAM2")
            vis_count["DINO"] += 1

        # Baseline 2: CLIPSeg
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        pred_p_c, pred_v_c = baseline_clipseg(image, gt_person.shape)
        elapsed = time.time() - start
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        stats["CLIPSeg"]["time"] += elapsed
        stats["CLIPSeg"]["samples"] += 1
        stats["CLIPSeg"]["mem"].append(mem_used)

        dice_clip_person.append(dice_score(pred_p_c, gt_person))
        dice_clip_vehicle.append(dice_score(pred_v_c, gt_vehicle))
        if vis_count["CLIPSeg"] < vis_limit:
            visualize_and_save(image, gt_person, gt_vehicle,
                               pred_p_c, pred_v_c,
                               "CLIPSeg", img_name,
                               "Outputs/Baseline_CLIPSeg")
            vis_count["CLIPSeg"] += 1

        # Proposed: YOLO + SAM2
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        pred_p_p, pred_v_p = proposed_yolo_sam2(image)
        elapsed = time.time() - start
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        stats["YOLO"]["time"] += elapsed
        stats["YOLO"]["samples"] += 1
        stats["YOLO"]["mem"].append(mem_used)

        dice_prop_person.append(dice_score(pred_p_p, gt_person))
        dice_prop_vehicle.append(dice_score(pred_v_p, gt_vehicle))
        if vis_count["YOLO"] < vis_limit:
            visualize_and_save(image, gt_person, gt_vehicle,
                               pred_p_p, pred_v_p,
                               "YOLOv8+SAM2", img_name,
                               "Outputs/Baseline_Yolov8_SAM2")
            vis_count["YOLO"] += 1

    # ==== RESULTS ====
    print("\n==== RESULTS (Mean Dice) ====")
    print(f"Baseline (DINO+SAM2): Person={np.mean(dice_dino_person):.3f}, Vehicle={np.mean(dice_dino_vehicle):.3f}")
    print(f"Baseline (CLIPSeg):   Person={np.mean(dice_clip_person):.3f}, Vehicle={np.mean(dice_clip_vehicle):.3f}")
    print(f"Proposed (YOLO+SAM2): Person={np.mean(dice_prop_person):.3f}, Vehicle={np.mean(dice_prop_vehicle):.3f}")

    # ==== TIMING & MEMORY RESULTS ====
    print("\n==== PERFORMANCE METRICS ====")
    for method, d in stats.items():
        avg_time = d["time"] / max(1, d["samples"])
        avg_mem = np.mean(d["mem"]) if d["mem"] else 0
        print(f"{method}:")
        print(f"  Total time   = {d['time']:.2f} s")
        print(f"  Time/sample  = {avg_time*1000:.2f} ms")
        print(f"  GPU memory   = {avg_mem:.2f} MB per sample")


if __name__ == "__main__":
    main()
