import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
print("Path to dataset files:", path)

# Local target directory
target_dir = os.path.join(os.getcwd(), "data/coco-2017-dataset")

# Copy dataset to local directory
if not os.path.exists(target_dir):
    shutil.copytree(path, target_dir)
    print(f"Dataset copied to: {target_dir}")
else:
    print(f"Dataset already exists at: {target_dir}")