"""
Fine‑tune CLIP, A‑CLIP and SigLIP models on the Human Action Recognition (HAR) dataset.

This script implements a complete end‑to‑end training pipeline for image
classification using three different vision–language backbones:

* **CLIP** – the standard OpenAI image encoder with a classification head on top.
* **SigLIP** – a model that replaces CLIP's contrastive softmax loss with a
  pairwise sigmoid objective during pretraining; here we use its
  ``SiglipForImageClassification`` variant which already includes a
  classification head.

The code is organised into functions to make it easy to train any of
the three backbones on the Human Action Recognition dataset.  The dataset
is assumed to be stored in the following layout:

```
<data_root>/train/<class_0>/*.jpg
<data_root>/train/<class_1>/*.jpg
 …
<data_root>/test/<class_0>/*.jpg
<data_root>/test/<class_1>/*.jpg
 …
```

where ``class_i`` are the action categories (e.g. ``calling``, ``clapping``,
``cycling``) and each folder contains JPEG (or PNG) frames for that class.

The script uses the **🤗 HuggingFace Transformers** and **datasets**
libraries to handle data loading, preprocessing, model initialisation and
training.  It also logs basic metrics such as accuracy and training time.

Example usage (run from the command line):

```
python fine_tune_har.py \
    --data_root /home/ashmal/Courses/CVS/Assignment_1/data/HumanActionRecognition/Structured \
    --model_name clip \
    --output_dir ./clip_har_finetuned \
    --num_train_epochs 6 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 5e-5

python fine_tune_har.py \
    --data_root /home/ashmal/Courses/CVS/Assignment_1/data/HumanActionRecognition/Structured \
    --model_name siglip \
    --output_dir ./siglip_har_finetuned \
    --num_train_epochs 6 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-6
```

Depending on your GPU memory, you may increase ``per_device_train_batch_size``
to much larger values.  In particular, the SigLIP backbone scales well to
large batch sizes (32 k and above【887825966266497†L975-L985】), whereas the
CLIP and A‑CLIP variants typically saturate around a few thousand samples
per device【887825966266497†L975-L985】.

Reference documentation:

* The **CLIP** image classification model provided by HuggingFace is
  ``CLIPForImageClassification``【671592583321384†L1628-L1700】.  It consists of a
  vision transformer backbone with a linear classification head on top.
* The **SigLIP** image classification model is available via
  ``SiglipForImageClassification``【887825966266497†L944-L959】; the accompanying
  ``AutoImageProcessor`` handles image resizing and normalisation.

Note that this script does not automatically upload the fine‑tuned models
to the Hugging Face Hub.  If desired, use ``trainer.save_model()`` and
``huggingface_hub`` APIs after training.
"""

import argparse
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import (
    AutoImageProcessor,
    CLIPForImageClassification,
    SiglipForImageClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_har_dataset(data_root: str) -> DatasetDict:
    """Load the Human Action Recognition dataset using the HuggingFace datasets API.

    The dataset is expected to reside under ``data_root`` with ``train`` and
    ``test`` subdirectories, each containing one folder per class.  The
    ``imagefolder`` loader automatically infers labels from subfolder names.

    Args:
        data_root: Path to the dataset root (should contain ``train`` and
            ``test`` directories).

    Returns:
        A ``DatasetDict`` with ``train`` and ``test`` splits and features
        ``image`` (PIL.Image) and ``label`` (ClassLabel).
    """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Expected to find 'train' and 'test' directories under {data_root}. "
            "Please verify that the dataset is organised as described in the README."
        )
    logger.info("Loading train dataset from %s", train_dir)
    train_ds = load_dataset("imagefolder", data_dir=train_dir, split="train")
    logger.info("Loading test dataset from %s", test_dir)
    test_ds = load_dataset("imagefolder", data_dir=test_dir, split="train")
    return DatasetDict({"train": train_ds, "test": test_ds})


def get_label_mappings(dataset: DatasetDict) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return mappings between integer IDs and string labels from the dataset."""
    # The imagefolder loader uses a ClassLabel feature for the ``label`` field.
    class_label = dataset["train"].features["label"]
    id2label = {i: name for i, name in enumerate(class_label.names)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def build_transforms(processor, train: bool) -> Compose:
    """Create a torchvision transform pipeline for training or evaluation.

    The HuggingFace image processors (e.g. ``AutoImageProcessor`` for CLIP and
    SigLIP) expose ``image_mean``, ``image_std`` and ``size`` attributes that we
    reuse here.

    Args:
        processor: An image processor instance from HuggingFace.
        train: Whether to include data augmentation (random horizontal flip) or
            not.  You can extend this with more augmentations if desired.

    Returns:
        A ``Compose`` object that maps a PIL image to a PyTorch tensor in
        channel‑first format with normalisation applied.
    """
    # Determine target size (height == width) from processor
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    image_mean = processor.image_mean if hasattr(processor, "image_mean") else [0.5, 0.5, 0.5]
    image_std = processor.image_std if hasattr(processor, "image_std") else [0.5, 0.5, 0.5]
    transforms: List[Any] = [Resize((size, size)), ToTensor(), Normalize(mean=image_mean, std=image_std)]
    return Compose(transforms)


@dataclass
class HARDataCollator:
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = []
        for ex in examples:
            pv = ex["pixel_values"]
            if isinstance(pv, list):
                pv = torch.tensor(pv)
            pixel_values.append(pv)
        pixel_values = torch.stack(pixel_values)
        labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean().item()


def preprocess_dataset(dataset: Dataset, transform: Compose) -> Dataset:
    """Apply the image transformation pipeline to a HuggingFace Dataset.

    The ``imagefolder`` loader returns PIL images; this function maps them
    through a torchvision transform to obtain PyTorch tensors.  The resulting
    dataset contains ``pixel_values`` (tensor) and ``label`` (int) keys.
    """
    def _transform(example: Dict[str, Any]) -> Dict[str, Any]:
        image = example["image"]
        tensor = transform(image.convert("RGB"))
        # torchvision transforms usually return a Tensor already, but force cast
        tensor = torch.as_tensor(tensor)
        example["pixel_values"] = tensor
        return example


    return dataset.map(_transform, remove_columns=["image"], batched=False)


def train_model(
    model_name: str,
    data_root: str,
    output_dir: str,
    num_train_epochs: int = 6,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 8,
    logging_steps: int = 50,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    base_model_name: Optional[str] = None,
) -> None:
    """Fine‑tune a model on the HAR dataset.

    Args:
        model_name: One of ``"clip"``, ``"siglip"``.  Determines
            which backbone to use.
        data_root: Path to the dataset root (train/test splits).
        output_dir: Directory where the fine‑tuned model and checkpoints will be
            saved.
        num_train_epochs: Number of epochs to train.
        learning_rate: Optimiser learning rate.
        per_device_train_batch_size: Batch size per GPU/CPU during training.
        per_device_eval_batch_size: Batch size per GPU/CPU during evaluation.
        logging_steps: Interval (in steps) at which to log metrics.
        evaluation_strategy: When to perform evaluation (e.g. ``"epoch"`` or
            ``"steps"``).
        save_strategy: When to save model checkpoints.
        warmup_steps: Number of warmup steps for the learning rate scheduler.
        weight_decay: Weight decay for AdamW optimiser.
        num_workers: Number of subprocesses used for data loading.
        base_model_name: Override the pretrained checkpoint name (optional).
    """

    dataset = load_har_dataset(data_root)
    id2label, label2id = get_label_mappings(dataset)
    num_labels = len(id2label)

    if model_name == "clip":
        # Use OpenAI CLIP vision backbone with a classification head
        pretrained_name = base_model_name or "openai/clip-vit-base-patch32"
        model = CLIPForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        processor = AutoImageProcessor.from_pretrained(pretrained_name)
    elif model_name == "siglip":
        # Use SigLIP classification model
        pretrained_name = base_model_name or "google/siglip-base-patch16-224"
        model = SiglipForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        processor = AutoImageProcessor.from_pretrained(pretrained_name)
    else:
        raise ValueError("Unsupported model_name. Choose from 'clip', 'siglip'.")

    # Build data transforms and preprocess datasets
    train_transform = build_transforms(processor, train=True)
    val_transform = build_transforms(processor, train=False)
    processed_train = preprocess_dataset(dataset["train"], train_transform)
    processed_eval = preprocess_dataset(dataset["test"], val_transform)

    # Use HuggingFace Trainer for CLIP and SigLIP
    if model_name in {"clip", "siglip"}:
        data_collator = HARDataCollator()
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_eval,
            data_collator=data_collator,
            tokenizer=processor,
            compute_metrics=compute_metrics,
        )

        start_time = time.time()
        # Perform evaluation before training to get baseline
        eval_metrics_pre = trainer.evaluate()
        logger.info("Initial evaluation accuracy: %.4f", eval_metrics_pre.get("eval_accuracy", 0.0))
        # Train
        trainer.train()
        elapsed = time.time() - start_time
        logger.info("Training completed in %.2f seconds", elapsed)
        # Final evaluation
        eval_metrics = trainer.evaluate()
        logger.info("Final evaluation accuracy: %.4f", eval_metrics.get("eval_accuracy", 0.0))
        # Save model
        trainer.save_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune vision–language models on the HAR dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the HAR dataset root")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["clip", "siglip"],
        required=True,
        help="Which model to fine‑tune: 'clip' (CLIPForImageClassification), 'siglip' (SiglipForImageClassification)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the fine‑tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per device")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N optimisation steps")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="When to evaluate the model")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="When to save model checkpoints")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimiser")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--base_model_name", type=str, default=None, help="Optional: override the pretrained checkpoint name")
    args = parser.parse_args()
    train_model(
        model_name=args.model_name,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        base_model_name=args.base_model_name,
    )


if __name__ == "__main__":
    main()