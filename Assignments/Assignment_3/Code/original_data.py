from pathlib import Path
import shutil

# Paths from GitHub repo structure
repo_root = Path("/home/ashmal/Courses/CVS/Assignment_3/CamVid")
img_repo_dir = repo_root / "CamVid_RGB"
mask_repo_dir = repo_root / "CamVid_Label"
val_list_file = repo_root / "camvid_val.txt"

# Output folder (new, for clean separation)
out_dir = Path("/home/ashmal/Courses/CVS/Assignment_3/data/CamVid_github")
val_out = out_dir / "val"
mask_out = out_dir / "val_labels"
val_out.mkdir(parents=True, exist_ok=True)
mask_out.mkdir(parents=True, exist_ok=True)

# Read validation list
with open(val_list_file, "r") as f:
    lines = f.read().splitlines()

# Each line in camvid_val.txt looks like:
for line in lines:
    img_rel, mask_rel = line.split()

    img_src = repo_root / img_rel
    mask_src = repo_root / mask_rel

    img_dst = val_out / Path(img_rel).name
    mask_dst = mask_out / Path(mask_rel).name

    if not img_src.exists():
        print(f"⚠️ Missing image: {img_src}")
        continue
    if not mask_src.exists():
        print(f"⚠️ Missing mask: {mask_src}")
        continue

    shutil.copy(img_src, img_dst)
    shutil.copy(mask_src, mask_dst)

print(f"✅ Validation set prepared under {val_out} and {mask_out}")
