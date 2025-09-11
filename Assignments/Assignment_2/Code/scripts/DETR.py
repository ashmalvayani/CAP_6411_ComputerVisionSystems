# Assignment#02
# Name: Ashmal Vayani
# UCF-ID: 5669011
# NIC: as193218

# Inference-only DETR (facebookresearch torch.hub) on COCO 2017 val.
# Metrics: COCO mAP (pycocotools) + Detection Confusion Matrix + P/R/F1 at chosen IoU/score.

import os
import argparse
import json
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import (
    collate_fn, xyxy_to_xywh, measure_text,
    update_confusion_matrix, prf1_from_counts, plot_confusion
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def detr_hub_postprocess(outputs: Dict[str, torch.Tensor], images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    pred_logits = outputs["pred_logits"].softmax(-1)[..., :-1]
    pred_boxes  = outputs["pred_boxes"]
    B, Q, _ = pred_boxes.shape
    results = []

    for i in range(B):
        # scores/labels
        scores, labels = pred_logits[i].max(-1)
        _, H, W = images[i].shape
        # cxcywh -> xyxy in pixels
        cx, cy, w, h = pred_boxes[i].unbind(-1)
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        x2 = (cx + 0.5 * w) * W
        y2 = (cy + 0.5 * h) * H
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        results.append({
            "boxes": boxes_xyxy,
            "scores": scores,
            "labels": labels,   # NOTE: DETR hub labels follow COCO IDs order with 0='N/A'
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="DETR (hub) COCO2017 val evaluation")
    parser.add_argument("--data-root", type=str, default="/home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017")
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.50)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--vis-samples", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="Output/DETR")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    img_dir = os.path.join(args.data_root, "val2017")
    ann = args.ann_file or os.path.join(args.data_root, "annotations", "instances_val2017.json")

    assert os.path.isdir(img_dir), f"val2017 not found: {img_dir}"
    assert os.path.isfile(ann), f"Annotation JSON not found: {ann}"

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "Sample_Outputs"), exist_ok=True)

    # Dataset / Loader
    transform = transforms.ToTensor()
    dataset = CocoDetection(root=img_dir, annFile=ann, transform=transform)
    if args.max_images and args.max_images > 0 and args.max_images < len(dataset):
        dataset.ids = dataset.ids[:args.max_images]
    print(f"Loaded COCO val2017: {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(device)
    model.eval()
    print("Loaded DETR hub model (pretrained on COCO).")

    # COCO handles
    cocoGt = COCO(ann)

    # Categories / confusion matrix setup
    cat_ids = cocoGt.getCatIds()
    cats = cocoGt.loadCats(cat_ids)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_name = [cat_id_to_name[cid] for cid in cat_ids]
    bg_idx = len(cat_ids)
    labels_with_bg = idx_to_name + ["_background_"]
    conf_mat = np.zeros((len(cat_ids) + 1, len(cat_ids) + 1), dtype=np.int64)

    # Inference loop (+ timing)
    results_json = []
    processed_img_ids = []
    print("Running inference...")

    wall_t0 = time.perf_counter()
    infer_time_total = 0.0
    total_images = 0

    with torch.no_grad():
        img_global_idx = 0
        for images, _targets in tqdm(loader):
            # move and normalize for DETR hub
            images_dev = [im.to(device, non_blocking=True) for im in images]
            mean = IMAGENET_MEAN.to(device)
            std = IMAGENET_STD.to(device)
            images_norm = [(im - mean) / std for im in images_dev]

            # forward pass timing (sync CUDA for accurate measurement)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(images_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time_total += (time.perf_counter() - t0)

            std_outs = detr_hub_postprocess(outputs, images)  # back to tv-style dicts
            bs = len(std_outs)
            total_images += bs

            for out in std_outs:
                image_id = dataset.ids[img_global_idx]
                img_global_idx += 1
                processed_img_ids.append(int(image_id))

                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()

                # DETR hub has 0='N/A'. Drop those and any labels not in COCO detection cat_ids.
                keep_valid = (labels != 0) & np.isin(labels, np.array(cat_ids))
                boxes = boxes[keep_valid]
                scores = scores[keep_valid]
                labels = labels[keep_valid]

                # For COCOeval: keep ALL remaining detections
                for box, score, label in zip(boxes, scores, labels):
                    results_json.append({
                        "image_id": int(image_id),
                        "category_id": int(label),  # already COCO IDs order
                        "bbox": xyxy_to_xywh(box),
                        "score": float(score),
                    })

                # For confusion matrix: filter by score, then IoU match
                keep = scores >= args.score_thr
                pb, ps, pl = boxes[keep], scores[keep], labels[keep]

                # Build GT arrays (all classes) for this image
                ann_ids = cocoGt.getAnnIds(imgIds=[image_id], iscrowd=None)
                gts = cocoGt.loadAnns(ann_ids)
                gt_boxes_all, gt_labels_all = [], []
                for g in gts:
                    if "bbox" not in g:
                        continue
                    x, y, w, h = g["bbox"]
                    gt_boxes_all.append([x, y, x + w, y + h])
                    gt_labels_all.append(g["category_id"])
                gt_boxes_all = np.array(gt_boxes_all, dtype=np.float32) if len(gt_boxes_all) else np.zeros((0, 4), np.float32)
                gt_labels_all = np.array(gt_labels_all, dtype=np.int64) if len(gt_labels_all) else np.zeros((0,), np.int64)

                update_confusion_matrix(
                    conf_mat, pb, ps, pl, gt_boxes_all, gt_labels_all,
                    args.iou_thr, cat_id_to_idx, bg_idx
                )

    # Save detections JSON
    det_path = os.path.join(args.outdir, "detr_test_predictions.json")
    with open(det_path, "w") as f:
        json.dump(results_json, f)
    print(f"Saved detections -> {det_path}")

    # COCO mAP (restricted if --max-images used)
    cocoDt = cocoGt.loadRes(results_json)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    if len(processed_img_ids) > 0:
        cocoEval.params.imgIds = sorted(set(processed_img_ids))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    mAP  = cocoEval.stats[0]   # AP@[.50:.95]
    AP50 = cocoEval.stats[1]
    AP75 = cocoEval.stats[2]
    AP_s, AP_m, AP_l = cocoEval.stats[3], cocoEval.stats[4], cocoEval.stats[5]
    AR1, AR10, AR100 = cocoEval.stats[6], cocoEval.stats[7], cocoEval.stats[8]
    AR_s, AR_m, AR_l = cocoEval.stats[9], cocoEval.stats[10], cocoEval.stats[11]

    # Micro P/R/F1 from confusion matrix
    tp = int(np.trace(conf_mat[:-1, :-1]))
    fp = int(conf_mat[bg_idx, :-1].sum())
    fn = int(conf_mat[:-1, bg_idx].sum())
    micro_p, micro_r, micro_f1 = prf1_from_counts(tp, fp, fn)
    print(f"[Fixed @ IoU={args.iou_thr:.2f}, score>={args.score_thr:.2f}] "
          f"P={micro_p:.4f}, R={micro_r:.4f}, F1={micro_f1:.4f}")

    # Timing summary
    wall_total = time.perf_counter() - wall_t0
    n = max(total_images, 1)
    infer_avg = infer_time_total / n
    wall_avg  = wall_total / n
    infer_throughput = n / infer_time_total if infer_time_total > 0 else 0.0
    wall_throughput  = n / wall_total if wall_total > 0 else 0.0

    print(f"\nTiming:")
    print(f"  Images processed: {total_images}")
    print(f"  Total time: {wall_total:.2f} s  |  {wall_throughput:.2f} img/s")
    print(f"  Model inference time (sum): {infer_time_total:.2f} s  |  {infer_throughput:.2f} img/s")
    print(f"  Avg. time per image: {wall_avg*1000:.2f} ms/img,  model-only: {infer_avg*1000:.2f} ms/img\n")

    # Save metrics + timing
    eval_txt = os.path.join(args.outdir, "detr_test_eval.txt")
    with open(eval_txt, "w") as f:
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
        f.write(f"Images processed: {total_images}\n")
        f.write(f"Total wall time (s): {wall_total:.3f}\n")
        f.write(f"Model inference time total (s): {infer_time_total:.3f}\n")
        f.write(f"Avg wall time per image (ms): {wall_avg*1000:.3f}\n")
        f.write(f"Avg model time per image (ms): {infer_avg*1000:.3f}\n")
        f.write(f"Throughput wall (img/s): {wall_throughput:.3f}\n")
        f.write(f"Throughput model-only (img/s): {infer_throughput:.3f}\n")
    print(f"Saved metrics -> {eval_txt}")

    # Save confusion matrix CSVs + heatmap
    cm_counts_csv = os.path.join(args.outdir, "detr_confusion_matrix_counts.csv")
    pd.DataFrame(conf_mat, index=labels_with_bg, columns=labels_with_bg).to_csv(cm_counts_csv, index=True)
    print(f"Saved confusion matrix counts -> {cm_counts_csv}")

    row_sums = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = np.divide(conf_mat, np.maximum(row_sums, 1), where=row_sums > 0)
    cm_norm_csv = os.path.join(args.outdir, "detr_confusion_matrix_row_normalized.csv")
    pd.DataFrame(cm_norm, index=labels_with_bg, columns=labels_with_bg).to_csv(cm_norm_csv, float_format="%.4f")
    print(f"Saved row-normalized confusion matrix -> {cm_norm_csv}")

    cm_png = os.path.join(args.outdir, "detr_confusion_matrix.png")
    plot_confusion(
        cm_norm, labels_with_bg, cm_png,
        title=f"DETR (hub) Confusion Matrix (IoU≥{args.iou_thr}, score≥{args.score_thr})"
    )
    print(f"Saved confusion matrix heatmap -> {cm_png}")

    # --------- Visualizations ---------
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
            # normalize single image before forward
            img_dev = img_t.to(device)
            img_dev = (img_dev - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            out = model([img_dev])
            out = detr_hub_postprocess(out, [img_t])[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        keep_valid = (labels != 0) & np.isin(labels, np.array(cat_ids))
        boxes, scores, labels = boxes[keep_valid], scores[keep_valid], labels[keep_valid]

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

        out_path = os.path.join(args.outdir, "Sample_Outputs", f"detr_{img_id}.jpg")
        im.save(out_path)

    print(f"Done. Visualizations in {os.path.join(args.outdir, 'Sample_Outputs')}")

if __name__ == "__main__":
    main()
