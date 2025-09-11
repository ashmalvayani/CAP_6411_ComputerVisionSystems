# Assignment#02
# Name: Ashmal Vayani
# UCF-ID: 5669011
# NIC: as193218

# Inference-only DINO evaluation on COCO 2017 val set.
# Metrics: COCO mAP (pycocotools) + Detection Confusion Matrix + P/R/F1 at chosen IoU/score.

"""
To run:
cd Assignment_2/scripts
python DINO_eval.py \
  --dino-repo /home/ashmal/Courses/CVS/Assignment_2/DINO \
  --config    DINO/configs/DINO/DINO_4scale.py \
  --checkpoint /home/ashmal/Courses/CVS/Assignment_2/DINO/Weights/checkpoint0011_4scale.pth \
  --data-root /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017 \
  --max-images 100 --outdir Output/DINO
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==== Our utilities ====
from utils import (
    collate_fn, xyxy_to_xywh, measure_text,
    update_confusion_matrix, prf1_from_counts, plot_confusion
)

# ==== Add repo to path ====
def add_repo_to_path(dino_repo):
    if dino_repo not in sys.path:
        sys.path.insert(0, dino_repo)

# ==== Load DINO ====
def load_dino(config_file, checkpoint_file, device):
    from DINO.util.slconfig import SLConfig
    from DINO.models import build_model

    args = SLConfig.fromfile(config_file)
    args.device = device
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    return model, postprocessors


def main():
    # Start wall clock
    t0_all = time.perf_counter()
    wall_start_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- Arguments ----
    parser = argparse.ArgumentParser(description="DINO COCO2017 val evaluation (inference only)")
    parser.add_argument("--dino-repo", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.50)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--vis-samples", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="Output/DINO")
    args = parser.parse_args()

    add_repo_to_path(args.dino_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Data ----
    img_dir = os.path.join(args.data_root, "val2017")
    ann = args.ann_file or os.path.join(args.data_root, "annotations", "instances_val2017.json")
    assert os.path.isdir(img_dir), f"val2017 folder not found: {img_dir}"
    assert os.path.isfile(ann), f"Annotation JSON not found: {ann}"

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "Sample_Outputs"), exist_ok=True)

    transform = transforms.ToTensor()
    dataset = CocoDetection(root=img_dir, annFile=ann, transform=transform)
    if args.max_images > 0 and args.max_images < len(dataset):
        dataset.ids = dataset.ids[:args.max_images]
    print(f"Loaded COCO val2017: {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # ---- Model ----
    model, postprocessors = load_dino(args.config, args.checkpoint, device)
    print("Loaded DINO model from:", args.checkpoint)

    # ---- COCO setup ----
    cocoGt = COCO(ann)
    cat_ids = cocoGt.getCatIds()
    cats = cocoGt.loadCats(cat_ids)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_name = [cat_id_to_name[cid] for cid in cat_ids]
    bg_idx = len(cat_ids)
    labels_with_bg = idx_to_name + ["_background_"]
    conf_mat = np.zeros((len(cat_ids) + 1, len(cat_ids) + 1), dtype=np.int64)

    # ---- Inference ----
    results_json = []
    processed_img_ids = []
    print("Running inference...")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0_infer = time.perf_counter()

    with torch.no_grad():
        img_global_idx = 0
        for images, _targets in tqdm(loader):
            for img in images:
                img = img.unsqueeze(0).to(device)
                out = model(img)
                outputs = postprocessors['bbox'](
                    out, torch.tensor(img.shape[-2:]).unsqueeze(0).to(device)
                )[0]

                image_id = dataset.ids[img_global_idx]
                img_global_idx += 1
                processed_img_ids.append(int(image_id))

                boxes = outputs["boxes"].cpu().numpy()
                scores = outputs["scores"].cpu().numpy()
                labels = outputs["labels"].cpu().numpy()

                # JSON for COCOeval
                for box, score, label in zip(boxes, scores, labels):
                    results_json.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": xyxy_to_xywh(box),
                        "score": float(score),
                    })

                # GT for confusion matrix
                ann_ids = cocoGt.getAnnIds(imgIds=[image_id], iscrowd=None)
                gts = cocoGt.loadAnns(ann_ids)
                gt_boxes_all, gt_labels_all = [], []
                for g in gts:
                    if "bbox" not in g:
                        continue
                    x, y, w, h = g["bbox"]
                    gt_boxes_all.append([x, y, x + w, y + h])
                    gt_labels_all.append(g["category_id"])
                gt_boxes_all = np.array(gt_boxes_all, dtype=np.float32) if gt_boxes_all else np.zeros((0, 4), np.float32)
                gt_labels_all = np.array(gt_labels_all, dtype=np.int64) if gt_labels_all else np.zeros((0,), np.int64)

                # Confusion matrix update
                keep = scores >= args.score_thr
                pb, ps, pl = boxes[keep], scores[keep], labels[keep]
                update_confusion_matrix(conf_mat, pb, ps, pl, gt_boxes_all, gt_labels_all,
                                        args.iou_thr, cat_id_to_idx, bg_idx)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1_infer = time.perf_counter()
    infer_seconds = t1_infer - t0_infer

    # ---- Save predictions ----
    det_path = os.path.join(args.outdir, "dino_test_predictions.json")
    with open(det_path, "w") as f:
        json.dump(results_json, f)
    print(f"Saved detections -> {det_path}")

    # ---- COCO Evaluation ----
    cocoDt = cocoGt.loadRes(results_json)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    if processed_img_ids:
        cocoEval.params.imgIds = sorted(set(processed_img_ids))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP, AP50, AP75 = cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[2]
    AP_s, AP_m, AP_l = cocoEval.stats[3], cocoEval.stats[4], cocoEval.stats[5]
    AR1, AR10, AR100 = cocoEval.stats[6], cocoEval.stats[7], cocoEval.stats[8]
    AR_s, AR_m, AR_l = cocoEval.stats[9], cocoEval.stats[10], cocoEval.stats[11]

    # ---- Micro P/R/F1 ----
    tp = int(np.trace(conf_mat[:-1, :-1]))
    fp = int(conf_mat[bg_idx, :-1].sum())
    fn = int(conf_mat[:-1, bg_idx].sum())
    micro_p, micro_r, micro_f1 = prf1_from_counts(tp, fp, fn)
    print(f"[Fixed @ IoU={args.iou_thr:.2f}, score>={args.score_thr:.2f}] "
          f"P={micro_p:.4f}, R={micro_r:.4f}, F1={micro_f1:.4f}")

    # ---- Timing ----
    num_imgs = len(processed_img_ids) if processed_img_ids else len(dataset)
    per_image_sec = (infer_seconds / num_imgs) if num_imgs > 0 else float("nan")
    imgs_per_sec = (num_imgs / infer_seconds) if infer_seconds > 0 else float("nan")

    t1_all = time.perf_counter()
    wall_seconds = t1_all - t0_all
    wall_end_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n--- Timing ---")
    print(f"Start time (local): {wall_start_iso}")
    print(f"End time   (local): {wall_end_iso}")
    print(f"Inference-only time: {infer_seconds:.3f} s")
    print(f"Images processed:    {num_imgs}")
    print(f"Per-image latency:   {per_image_sec*1000:.3f} ms/image")
    print(f"Throughput:          {imgs_per_sec:.3f} images/s")
    print(f"End-to-end walltime: {wall_seconds:.3f} s\n")

    # ---- Save metrics ----
    eval_txt = os.path.join(args.outdir, "dino_test_eval.txt")
    with open(eval_txt, "w") as f:
        f.write(f"Run timestamps (local):\n")
        f.write(f"  Start: {wall_start_iso}\n")
        f.write(f"  End:   {wall_end_iso}\n\n")

        f.write("COCO Evaluation Metrics (pycocotools):\n")
        f.write(f"mAP@[.50:.95]: {mAP:.4f}\n")
        f.write(f"AP50: {AP50:.4f}\n")
        f.write(f"AP75: {AP75:.4f}\n")
        f.write(f"AP (small/medium/large): {AP_s:.4f}/{AP_m:.4f}/{AP_l:.4f}\n")
        f.write(f"AR@1/10/100: {AR1:.4f}/{AR10:.4f}/{AR100:.4f}\n")
        f.write(f"AR (small/medium/large): {AR_s:.4f}/{AR_m:.4f}/{AR_l:.4f}\n\n")
        f.write(f"Fixed-threshold PRF1 (IoU={args.iou_thr:.2f}, score>={args.score_thr:.2f}):\n")
        f.write(f"Precision: {micro_p:.4f}\n")
        f.write(f"Recall:    {micro_r:.4f}\n")
        f.write(f"F1:        {micro_f1:.4f}\n")
        f.write(f"(Counts) TP={tp} FP={fp} FN={fn}\n\n")

        f.write("Timing:\n")
        f.write(f"Inference-only seconds: {infer_seconds:.3f}\n")
        f.write(f"Images processed:       {num_imgs}\n")
        f.write(f"Per-image seconds:      {per_image_sec:.6f}\n")
        f.write(f"Throughput (img/s):     {imgs_per_sec:.3f}\n")
        f.write(f"End-to-end seconds:     {wall_seconds:.3f}\n")

    print(f"Saved metrics -> {eval_txt}")

    # ---- Confusion Matrix ----
    cm_counts_csv = os.path.join(args.outdir, "dino_confusion_matrix_counts.csv")
    pd.DataFrame(conf_mat, index=labels_with_bg, columns=labels_with_bg)\
      .to_csv(cm_counts_csv, index=True)
    print(f"Saved confusion matrix counts -> {cm_counts_csv}")

    row_sums = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = np.divide(conf_mat, np.maximum(row_sums, 1), where=row_sums > 0)
    cm_norm_csv = os.path.join(args.outdir, "dino_confusion_matrix_row_normalized.csv")
    pd.DataFrame(cm_norm, index=labels_with_bg, columns=labels_with_bg)\
      .to_csv(cm_norm_csv, float_format="%.4f")
    print(f"Saved row-normalized confusion matrix -> {cm_norm_csv}")

    cm_png = os.path.join(args.outdir, "dino_confusion_matrix.png")
    plot_confusion(
        cm_norm,
        labels_with_bg,
        cm_png,
        title=f"DINO Confusion Matrix (IoU≥{args.iou_thr}, score≥{args.score_thr})"
    )
    print(f"Saved confusion matrix heatmap -> {cm_png}")

    # ---- Sample Visualizations ----
    print("Saving sample visualizations...")
    np.random.seed(123)
    font = ImageFont.load_default()
    sample_count = min(args.vis_samples, len(dataset))
    sample_idxs = np.random.choice(len(dataset), size=sample_count, replace=False).tolist()

    for idx in sample_idxs:
        img_t, _ = dataset[idx]
        im = transforms.ToPILImage()(img_t)
        img_id = dataset.ids[idx]

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
            outputs = postprocessors['bbox'](
                out, torch.tensor(img_t.shape[-2:]).unsqueeze(0).to(device)
            )[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        keep = scores >= args.score_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(scores) > 10:
            top = np.argsort(-scores)[:10]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        draw = ImageDraw.Draw(im)
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            name = cat_id_to_name.get(int(lb), str(int(lb)))
            txt = f"{name}:{sc:.2f}"
            tw, th = measure_text(draw, txt, font)
            ty = max(0, y1 - th)
            draw.rectangle([x1, ty, x1 + tw + 2, ty + th], fill="red")
            draw.text((x1 + 1, ty), txt, fill="yellow", font=font)

        out_path = os.path.join(args.outdir, "Sample_Outputs", f"dino_{img_id}.jpg")
        im.save(out_path)

    print(f"Done. Visualizations in {os.path.join(args.outdir, 'Sample_Outputs')}")

if __name__ == "__main__":
    main()
