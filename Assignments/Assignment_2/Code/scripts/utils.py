# utils.py
# Shared utilities for COCO detection evaluation & visualization.

from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

def collate_fn(batch):
    return tuple(zip(*batch))

def xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [Na, 4] xyxy
    b: [Nb, 4] xyxy
    returns IoU matrix [Na, Nb]
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ixmin = np.maximum(a[:, None, 0], b[None, :, 0])
    iymin = np.maximum(a[:, None, 1], b[None, :, 1])
    ixmax = np.minimum(a[:, None, 2], b[None, :, 2])
    iymax = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Return (width, height) for text
    """
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r-l, b-t)
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return (r-l, b-t)
    if hasattr(font, "getsize"):
        return font.getsize(text)
    return (len(text) * 6, 10)

def update_confusion_matrix(
    cm: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_thr: float,
    cat_id_to_idx: Dict[int, int],
    bg_idx: int,
):
    # sort predictions by confidence
    order = np.argsort(-pred_scores)
    p_boxes = pred_boxes[order]
    p_scores = pred_scores[order]
    p_labels = pred_labels[order]

    matched = np.zeros(len(gt_boxes), dtype=bool)

    if len(gt_boxes) > 0 and len(p_boxes) > 0:
        ious = iou_matrix(p_boxes, gt_boxes)
    else:
        ious = np.zeros((len(p_boxes), len(gt_boxes)), dtype=np.float32)

    for pi in range(len(p_boxes)):
        gi = np.argmax(ious[pi]) if len(gt_boxes) > 0 else -1
        if gi != -1 and ious[pi, gi] >= iou_thr and not matched[gi]:
            true_idx = cat_id_to_idx[int(gt_labels[gi])]
            pred_idx = cat_id_to_idx[int(p_labels[pi])]
            cm[true_idx, pred_idx] += 1
            matched[gi] = True
        else:
            pred_idx = cat_id_to_idx[int(p_labels[pi])]
            cm[bg_idx, pred_idx] += 1  # FP to background row

    # Unmatched GT -> FN (background column)
    for gi in range(len(gt_boxes)):
        if not matched[gi]:
            true_idx = cat_id_to_idx[int(gt_labels[gi])]
            cm[true_idx, bg_idx] += 1

def prf1_from_counts(tp, fp, fn) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def plot_confusion(cm: np.ndarray, labels: List[str], outpath: str, title: str):
    """Simple heatmap for (C+1)x(C+1) confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
