import argparse
import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    CLIPForImageClassification,
    SiglipForImageClassification,
)

"""
python visualize_output.py \
  --model_name clip \
  --model_dir ./clip_har_finetuned \
  --data_root /home/ashmal/Courses/CVS/Assignment_1/data/HumanActionRecognition/Structured \
  --out_dir SampleOutputs/CLIP \
  --num_examples 12

python visualize_output.py \
  --model_name siglip \
  --model_dir ./siglip_har_finetuned \
  --data_root /home/ashmal/Courses/CVS/Assignment_1/data/HumanActionRecognition/Structured \
  --out_dir SampleOutputs/SigLIP \
  --num_examples 12
"""

def visualize_predictions(model_name, model_dir, data_root, out_dir, num_examples=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=os.path.join(data_root, "test"), split="train")
    id2label = dataset.features["label"].names

    dataset = dataset.shuffle(seed=42).select(range(num_examples))

    # Load model + processor
    if model_name == "clip":
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = CLIPForImageClassification.from_pretrained(model_dir).to(device)
    elif model_name == "siglip":
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = SiglipForImageClassification.from_pretrained(model_dir).to(device)
    else:
        raise ValueError("model_name must be clip | siglip | aclip")

    model.eval()

    # Create output folder
    os.makedirs(out_dir, exist_ok=True)

    # Iterate and save each image separately
    for idx, example in enumerate(dataset):
        inputs = processor(images=example["image"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pred = logits.argmax(dim=-1).item()

        gt_label = id2label[example["label"]]
        pred_label = id2label[pred]

        plt.figure(figsize=(3, 3))
        plt.imshow(example["image"])
        plt.axis("off")
        plt.title(f"GT: {gt_label}\nPred: {pred_label}",
                  fontsize=10,
                  color="green" if gt_label == pred_label else "red")
        
        save_path = os.path.join(out_dir, f"{idx:03d}_{gt_label}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=["clip", "siglip"])
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True, help="Folder to save individual outputs")
    parser.add_argument("--num_examples", type=int, default=12)
    args = parser.parse_args()

    visualize_predictions(args.model_name, args.model_dir, args.data_root, args.out_dir, args.num_examples)
