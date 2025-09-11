# Assignment#01
# Name: Ashmal Vayani
# UCF-ID: 5669011
# NIC: as193218

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # For evaluation
import matplotlib.pyplot as plt

# Checking if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

path = "data/HumanActionRecognition"

# Define train and test directory paths
train_dir = os.path.join(path, "Structured", "train")
test_dir = os.path.join(path, "Structured", "test")

print("Train dir exists:", os.path.isdir(train_dir))
print("Test dir exists:", os.path.isdir(test_dir))

# Verifying classes in train and test set
print("Sample Classes in Train:", os.listdir(train_dir)[:5])
print("Total Classes in Train Set:", len(os.listdir(train_dir)))

print("Sample Classes in Test:", os.listdir(test_dir)[:5])
print("Total Classes in Test Set:", len(os.listdir(train_dir)))

## Data Loading and Preprocessing

# Define normalization means and stds using previous standard values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Transforms for training and validation/test
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # random crop to 224x224
    transforms.RandomHorizontalFlip(),      # random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# For validation and test, no random augmentation, just resize and center crop
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Load training dataset using ImageFolder
full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
num_classes = len(full_train_dataset.classes)
print("Detected classes:", full_train_dataset.classes)
print("Total training images:", len(full_train_dataset))

# Split into train and validation
val_ratio = 0.1
num_val = int(val_ratio * len(full_train_dataset))
num_train = len(full_train_dataset) - num_val

train_dataset, val_dataset = random_split(full_train_dataset, [num_train, num_val])
val_dataset.dataset.transform = test_transform # For validation, use the test_transform (no augmentation)

print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

# DataLoader for train and val
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Preparing test loader
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Test classes: {test_dataset.classes}")
print(f"Total test images: {len(test_dataset)}")

class_names = full_train_dataset.classes  # from train ImageFolder
assert test_dataset.classes == class_names, "Train/Test class orders differ!"

## Model Definition

vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_model = vit_model.to(device)

# Define loss and optimizer for ViT
criterion_vit = nn.CrossEntropyLoss()
optimizer_vit = optim.Adam(vit_model.parameters(), lr=1e-4)

## Training ViT

num_epochs = 50
best_val_acc_vit = 0.0
vit_log_file = open("Output/ViT/vit_training_log.txt", "w")

for epoch in tqdm(range(1, num_epochs+1)):
    vit_model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_vit.zero_grad()
        outputs = vit_model(images)
        loss = criterion_vit(outputs, labels)
        loss.backward()
        optimizer_vit.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    
    epoch_loss = running_loss / total_train
    epoch_acc = correct_train / total_train

    vit_model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
        
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = vit_model(images)
            loss = criterion_vit(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
    val_loss = val_loss / total_val
    val_acc = correct_val / total_val

    if val_acc > best_val_acc_vit:
        best_val_acc_vit = val_acc
        torch.save(vit_model.state_dict(), "Output/ViT/best_vit_model.pth")
    
    log_msg = (f"Epoch {epoch}/{num_epochs} - "
               f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - "
               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(log_msg)
    vit_log_file.write(log_msg + "\n")

vit_log_file.close()
print("ViT training complete. Best Val Acc: {:.4f}".format(best_val_acc_vit))


## Evaluation

# Load best ViT weights
vit_model.load_state_dict(torch.load("Output/ViT/best_vit_model.pth", map_location=device))
vit_model.eval()

class_names = full_train_dataset.classes

all_preds, all_labels, all_fnames = [], [], []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        logits = vit_model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Accumulate preds + GT
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

        # Recovering filenames for this batch from ImageFolder's samples (since shuffle=False)
        start = i * test_loader.batch_size
        end   = start + labels.size(0)
        batch_paths = [os.path.basename(p) for p, _ in test_dataset.samples[start:end]]
        all_fnames.extend(batch_paths)

# CSV of predictions
pred_labels_text = [class_names[i] for i in all_preds]
pred_df_vit = pd.DataFrame({"image": all_fnames, "label": pred_labels_text})
pred_df_vit.to_csv("Output/ViT/vit_test_predictions.csv", index=False)
print("Saved -> Output/ViT/vit_test_predictions.csv")
print(pred_df_vit.head(10))

# Evaluation Metrics: accuracy, classification report, confusion matrix
acc_vit = accuracy_score(all_labels, all_preds)
report_vit = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
cm_vit = confusion_matrix(all_labels, all_preds)
cm_vit_norm = cm_vit.astype(float) / cm_vit.sum(axis=1, keepdims=True)

# Save a text report
with open("Output/ViT/vit_test_eval.txt", "w") as f:
    f.write(f"Test Accuracy: {acc_vit:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_vit + "\n")
    f.write("Confusion Matrix (rows=true, cols=pred):\n")
    for row in cm_vit:
        f.write(" ".join(map(str, row)) + "\n")

print(f"ViT Test Accuracy: {acc_vit:.4f}")
print(report_vit)
print("Saved -> Output/ViT/vit_test_eval.txt")

# Ploting & saving confusion matrices
def plot_cm(matrix, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = f"{val:.2f}" if matrix.dtype.kind == 'f' else f"{val}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()

plot_cm(cm_vit, class_names, "ViT Confusion Matrix", "Output/ViT/vit_confusion_matrix.png")
plot_cm(cm_vit_norm, class_names, "ViT Confusion Matrix (Normalized)", "Output/ViT/vit_confusion_matrix_normalized.png")
print("Saved CM images in Output/ViT/")