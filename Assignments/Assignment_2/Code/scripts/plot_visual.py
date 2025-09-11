#!/usr/bin/env python3
"""
COCO-2017 Ground-Truth Visualizer (val set)

Usage examples:
  python plot_visual.py \
      --ann /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/annotations/instances_val2017.json \
      --img-dir /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/val2017 \
      --outdir gt_vis --n 50 --masks

  # Only persons and dogs, 12 images, shuffled
  python visualize_coco_gt.py --cats person,dog --n 12 --shuffle \
      --ann annotations/instances_val2017.json --img-dir val2017 --outdir gt_vis

Requirements: pycocotools, Pillow, matplotlib, numpy, tqdm
"""
import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.cm as cm

# ---------- utils ----------

def _mk_outdir(p: str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    # Try a few common fonts; fall back to default.
    for f in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def build_category_maps(coco: COCO) -> Tuple[Dict[int, str], Dict[int, Tuple[int,int,int]]]:
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {c["id"]: c["name"] for c in cats}
    # Deterministic color per category using tab20 colormap
    cmap = cm.get_cmap("tab20", len(cats) + 1)
    id_to_color = {}
    for i, c in enumerate(sorted(id_to_name.keys())):
        r, g, b, _ = cmap(i)
        id_to_color[c] = (int(r * 255), int(g * 255), int(b * 255))
    return id_to_name, id_to_color

def overlay_mask(img_rgba: Image.Image, mask: np.ndarray, color: Tuple[int,int,int], alpha: float = 0.35) -> Image.Image:
    """Alpha-overlay a binary mask onto img_rgba."""
    h, w = mask.shape
    overlay = Image.new("RGBA", (w, h), color + (int(alpha * 255),))
    # use mask as L image
    mask_l = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    masked_overlay = Image.composite(overlay, Image.new("RGBA", (w, h), (0,0,0,0)), mask_l)
    return Image.alpha_composite(img_rgba, masked_overlay)

def draw_box_and_label(draw: ImageDraw.ImageDraw, bbox, label: str, color: Tuple[int,int,int], font: ImageFont.ImageFont):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    # rectangle
    for t in range(max(1, int(round(min(w, h) * 0.01)))):
        draw.rectangle([x - t, y - t, x2 + t, y2 + t], outline=color, width=1)

    # label background
    text_w, text_h = draw.textbbox((0,0), label, font=font)[2:]
    pad = 2
    draw.rectangle([x, y - text_h - 2*pad, x + text_w + 2*pad, y], fill=color)
    draw.text((x + pad, y - text_h - pad), label, fill=(0,0,0), font=font)

def visualize_image(
    coco: COCO,
    image_id: int,
    img_dir: str,
    id_to_name: Dict[int, str],
    id_to_color: Dict[int, Tuple[int,int,int]],
    draw_masks: bool = False,
) -> Image.Image:
    img_info = coco.loadImgs([image_id])[0]
    img_path = os.path.join(img_dir, img_info["file_name"])
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(size=max(12, img.size[0] // 60))

    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    img_rgba = img.convert("RGBA")
    for ann in anns:
        cat_id = ann["category_id"]
        cat_name = id_to_name.get(cat_id, str(cat_id))
        color = id_to_color.get(cat_id, (255, 0, 0))

        # Optional mask overlay
        if draw_masks and "segmentation" in ann and ann["segmentation"]:
            try:
                mask = coco.annToMask(ann)
                img_rgba = overlay_mask(img_rgba, mask, color, alpha=0.35)
            except Exception:
                pass  # if malformed segmentation, just skip mask

        # Bounding box + label
        bbox = ann["bbox"]  # [x,y,w,h] in pixels
        label = f"{cat_name}"
        draw_box_and_label(ImageDraw.Draw(img_rgba), bbox, label, color + (255,), font)

    return img_rgba.convert("RGB")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Visualize COCO-2017 ground truth (val set).")
    ap.add_argument("--ann", required=True, help="Path to instances_val2017.json")
    ap.add_argument("--img-dir", required=True, help="Path to val2017 images directory")
    ap.add_argument("--outdir", default="gt_vis", help="Where to save visualizations")
    ap.add_argument("--n", type=int, default=16, help="How many images to render (<= total)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle image order")
    ap.add_argument("--masks", action="store_true", help="Overlay segmentation masks")
    ap.add_argument("--cats", type=str, default="", help="Comma-separated category names to include (optional)")
    args = ap.parse_args()

    coco = COCO(args.ann)
    id_to_name, id_to_color = build_category_maps(coco)

    # Optional category filtering
    cat_filter = []
    if args.cats:
        names = [c.strip() for c in args.cats.split(",") if c.strip()]
        cat_filter = coco.getCatIds(catNms=names)
        if not cat_filter:
            print(f"[WARN] No categories matched {names}; continuing without filter.")

    img_ids = coco.getImgIds()
    if args.shuffle:
        random.shuffle(img_ids)

    outdir = _mk_outdir(args.outdir)

    rendered = 0
    pbar = tqdm(img_ids, desc="Rendering GT", unit="img")
    for img_id in pbar:
        if cat_filter:
            # keep image only if it has at least one ann from the chosen categories
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_filter, iscrowd=None)
            if len(ann_ids) == 0:
                continue

        vis = visualize_image(
            coco, img_id, args.img_dir, id_to_name, id_to_color, draw_masks=args.masks
        )

        file_name = coco.loadImgs([img_id])[0]["file_name"]
        save_path = outdir / file_name
        vis.save(save_path, quality=95)
        rendered += 1
        pbar.set_postfix(saved=str(save_path.name))
        if rendered >= args.n:
            break

    print(f"Done. Saved {rendered} visualizations to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
