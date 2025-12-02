import os
from generate_dataset import get_prompts

image_dir = "sdxl_turbo_dataset"
caption_dir = "captions"
os.makedirs(caption_dir, exist_ok=True)

# Assume 'prompts' = list of 10 prompt strings, IDs 311–320
idxs = list(range(311, 321))
images = sorted(os.listdir(image_dir))
prompts = get_prompts()

# Map each image to its corresponding prompt based on its prefix (e.g. prompt311_)
for img in images:
    if not img.endswith(".png"): 
        continue
    num = int(img.replace("prompt", "").split("_")[0])  # extract 311–320
    prompt_text = prompts[idxs.index(num)]
    txt_path = os.path.join(caption_dir, img.replace(".png", ".txt"))
    with open(txt_path, "w") as f:
        f.write(prompt_text)
    print(f"Caption saved for {img}")
