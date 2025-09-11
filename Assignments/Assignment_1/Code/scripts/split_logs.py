import os
import re
import pandas as pd
import matplotlib.pyplot as plt


log_path = "resnet_training_log.txt"                 # change to resnet_training_log.txt for ResNet
out_dir  = "Output/ResNet-18"                           # change to Output/ResNet-18 for ResNet
os.makedirs(out_dir, exist_ok=True)

pattern = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s*-\s*"
    r"Train\s*Loss:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*"
    r"Train\s*Acc:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*-\s*"
    r"Val\s*Loss:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*"
    r"Val\s*Acc:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    flags=re.IGNORECASE
)

rows = []
with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epoch      = int(m.group(1))
            total_ep   = int(m.group(2))
            train_loss = float(m.group(3))
            train_acc  = float(m.group(4))
            val_loss   = float(m.group(5))
            val_acc    = float(m.group(6))
            rows.append({
                "epoch": epoch,
                "total_epochs": total_ep,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
epochs = df["epoch"]

def save_line(x, y, title, ylabel, filename):
    plt.figure(figsize=(7,4))
    plt.plot(x, y, marker='o', linewidth=1.8)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")

save_line(epochs, df["train_loss"], "ResNet-18: Training Loss", "Loss", "resnet_train_loss.png")
save_line(epochs, df["train_acc"],  "ResNet-18: Training Accuracy", "Accuracy", "resnet_train_acc.png")
save_line(epochs, df["val_loss"],   "ResNet-18: Validation Loss", "Loss", "resnet_val_loss.png")
save_line(epochs, df["val_acc"],    "ResNet-18: Validation Accuracy", "Accuracy", "resnet_val_acc.png")