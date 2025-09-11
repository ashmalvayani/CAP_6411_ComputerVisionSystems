import os
import csv
import argparse
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    p = argparse.ArgumentParser(
        description="Live demo from saved CSV predictions (ResNet-18 & ViT)."
    )
    p.add_argument("--test_dir", type=str, required=True,
                   help="Path to class-structured test folder (15 subfolders).")
    p.add_argument("--resnet_csv", type=str, required=True,
                   help="CSV with columns [image,label] for ResNet.")
    p.add_argument("--vit_csv", type=str, required=True,
                   help="CSV with columns [image,label] for ViT.")
    p.add_argument("--save_dir", type=str, default="demo_outputs_csv",
                   help="Where to save annotated images.")
    p.add_argument("--num", type=int, default=50, help="How many images to render.")
    return p.parse_args()

def load_predictions(csv_path):
    """
    Returns dict: filename -> predicted_label
    """
    pred = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # accept header names like image,label (your files use these)
        for row in reader:
            fname = row.get("image") or row.get("filename") or row.get("file")
            lab   = row.get("label") or row.get("pred") or row.get("prediction")
            if fname is None or lab is None:
                raise ValueError(f"CSV {csv_path} must have columns image,label")
            pred[fname] = lab
    return pred

def index_test_folder(test_dir):
    """
    Walk class-structured test folder and return:
      - files: list of filenames (e.g., 'Image_10910.jpg')
      - fname_to_gt: filename -> ground_truth_class
      - fname_to_path: filename -> absolute path
      - classes: sorted list of class names found
    """
    fname_to_gt = {}
    fname_to_path = {}
    classes = []
    for cls in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        classes.append(cls)
        for name in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                if name in fname_to_gt:
                    pass
                fname_to_gt[name] = cls
                fname_to_path[name] = os.path.join(cls_dir, name)
                
    file_list = sorted(fname_to_gt.keys(), key=lambda x: (x.split('.')[0].split('_')[-1].zfill(6), x))
    return file_list, fname_to_gt, fname_to_path, classes

def draw_overlay(pil_img, gt, rn_pred, vt_pred):
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    w, h = img.size
    banner_h = 90
    draw.rectangle([0, 0, w, banner_h], fill=(0, 0, 0, 200))

    def col(pred):  # green if correct, red otherwise
        return (0, 210, 0) if pred == gt else (230, 60, 60)

    draw.text((10, 8),  f"GT: {gt}",           fill=(230, 230, 230), font=font_title)
    draw.text((10, 36), f"ResNet-18: {rn_pred}", fill=col(rn_pred),    font=font_small)
    draw.text((10, 62), f"ViT-B/16: {vt_pred}",  fill=col(vt_pred),    font=font_small)
    return img

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Index test set ground truth
    files, fname_to_gt, fname_to_path, classes = index_test_folder(args.test_dir)
    print(f"Found {len(files)} test images across {len(classes)} classes.")

    # Load predictions
    rn_pred = load_predictions(args.resnet_csv)
    vt_pred = load_predictions(args.vit_csv)
    print(f"Loaded predictions: ResNet={len(rn_pred)}, ViT={len(vt_pred)}.")

    # Join and render
    saved = 0
    skipped_missing = 0
    for fname in files:
        if saved >= args.num:
            break
        if fname not in rn_pred or fname not in vt_pred:
            skipped_missing += 1
            continue
        gt = fname_to_gt[fname]
        rn = rn_pred[fname]
        vt = vt_pred[fname]

        # load image
        path = fname_to_path[fname]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skip unreadable: {path} ({e})")
            continue

        # Resize + center crop to 224x224 for a consistent visual (matches training)
        # (Optional) Only for nicer panels; the overlay works on any size.
        short_side = min(img.size)
        # simple center crop square
        left = (img.width  - short_side) // 2
        top  = (img.height - short_side) // 2
        img = img.crop((left, top, left + short_side, top + short_side))
        img = img.resize((224, 224), Image.BICUBIC)

        out = draw_overlay(img, gt, rn, vt)
        out_path = os.path.join(args.save_dir, f"{os.path.splitext(fname)[0]}_demo.png")
        out.save(out_path)

        ok_rn = "OK" if rn == gt else "XX"
        ok_vt = "OK" if vt == gt else "XX"
        print(f"[{saved+1:03d}] {fname} | GT={gt:18s} | RN={rn:18s} ({ok_rn}) | VT={vt:18s} ({ok_vt}) -> {out_path}")
        saved += 1

    print(f"Saved {saved} annotated frames to {args.save_dir} (skipped missing={skipped_missing}).")
    print("To produce a demo video:")
    print("  ffmpeg -y -framerate 4 -pattern_type glob -i 'live_demo_outputs/ResNet-18/*_demo.png' "
          "-c:v libx264 -pix_fmt yuv420p live_demo.mp4")

if __name__ == "__main__":
    main()
