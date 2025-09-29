# --- put this at the very top of finetune_internvl3_mathvista.py ---
import os

# 1) force single GPU and no display (avoids the X11 auth line)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.pop("DISPLAY", None)

# 2) nuke every SLURM/PMI/PMIX/OMPI variable (there are many)
for k in list(os.environ):
    if k.startswith(("OMPI_", "PMI_", "PMIX_", "SLURM_")):
        os.environ.pop(k, None)

# 3) make Accelerate/DeepSpeed pick NCCL and avoid MPI/PMI completely
os.environ["ACCELERATE_USE_DEEPSPEED"] = "1"
os.environ["ACCELERATE_USE_MPI"] = "false"
os.environ["DEEPSPEED_USE_MPI"] = "false"
os.environ["DEEPSPEED_COMM_BACKEND"] = "nccl"

# 4) don’t build DS CUDA ops (we already switched optimizer to AdamW)
os.environ["DS_BUILD_OPS"] = "0"

# 5) keep it 1 process / 1 GPU explicitly for safety
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ.setdefault("MASTER_PORT", "29500")
# --- end sanitizer ---


"""End‑to‑end LoRA fine‑tuning for InternVL3 on the MathVista dataset.

This script demonstrates how to fine‑tune a multimodal InternVL3 model on the
`AI4Math/MathVista` testmini split (1 k samples) using Low‑Rank
Adaptation (LoRA) and DeepSpeed ZeRO‑3.  It follows the recommended
workflow from InternVL’s official documentation: the vision encoder is
frozen and LoRA is applied only to the language backbone.  The MathVista
dataset contains questions, answers and images; see its dataset card
for a description of the fields【92535232541055†L87-L152】.  The model is loaded from
Hugging Face via `AutoModelForImageTextToText`.  When using a Hugging Face
native checkpoint (such as `OpenGVLab/InternVL3-1B-hf`), the model can be
loaded directly without specifying `trust_remote_code` or `use_flash_attn`.
Flash attention is selected automatically by the library when available.

To run this script you need a machine with at least one 40 GB GPU and
the following Python packages installed:

* PyTorch ≥ 2.2 with CUDA support.
* transformers ≥ 4.40 (for InternVL support and flash attention).
* datasets ≥ 2.18 (to stream MathVista from Hugging Face).
* peft ≥ 0.9 (for LoRA).
* deepspeed ≥ 0.13.0 (for ZeRO‑3 training).

Example usage:

```bash
# Install the required libraries.  A recent version of Transformers (≥ 4.52.0)
# is needed to load the InternVL3 "‑hf" checkpoints via AutoModelForImageTextToText.
pip install --upgrade "torch>=2.2" "transformers>=4.52.0" datasets peft deepspeed accelerate

# Fine‑tune InternVL3‑1B on the MathVista testmini split (1 k samples).
python finetune_internvl3_mathvista.py \
    --model_name OpenGVLab/InternVL3-1B-hf \
    --dataset_name AI4Math/MathVista \
    --subset testmini \
    --output_dir ./internvl3_mathvista_lora \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1
```

Notes:

* Only LoRA adapters are trained, so the GPU memory footprint remains
  modest.  However, the base InternVL3 model still uses ~20–30 GB of
  VRAM depending on its size.  Adjust the number of accumulation steps
  and batch size if you encounter out‑of‑memory errors.
* The script automatically writes a DeepSpeed configuration
  (`ds_config.json`) into the output directory.  You can tweak its
  offloading behaviour if you have multiple GPUs or more CPU RAM.
* If you see an error like “Unrecognized configuration class …” when
  loading a model, upgrade `transformers` to ≥ 4.52.0 and use a
  checkpoint ending with `-hf` (e.g. `OpenGVLab/InternVL3-1B-hf`).  These
  checkpoints include a native implementation of the model that is
  compatible with `AutoModelForImageTextToText`【569562641712897†L82-L117】.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="LoRA fine‑tuning for InternVL3 on MathVista")
    parser.add_argument(
        "--model_name",
        type=str,
        # Use the native Transformers implementation of InternVL3 by default.  The
        # "-hf" suffix indicates compatibility with the core HF library and
        # avoids configuration errors when using `AutoModelForImageTextToText`.
        default="OpenGVLab/InternVL3-1B-hf",
        help=(
            "Name or path of the InternVL3 checkpoint to fine‑tune.  Models ending "
            "with '-hf' are recommended (e.g. OpenGVLab/InternVL3-1B-hf) because "
            "they include native Transformers implementations and can be loaded "
            "with AutoModelForImageTextToText without relying on remote code."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="AI4Math/MathVista",
        help="Dataset identifier on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="testmini",
        help="Which split of MathVista to use (e.g. 'testmini' or 'default').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store checkpoints and DeepSpeed config.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Micro‑batch size per GPU.  Because images are large, 1 is a safe default.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.  Increase this to simulate a larger batch size.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (r). Lower values reduce memory usage but may limit performance.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA adapters.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (text tokens).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Where to report metrics (e.g. 'tensorboard', 'wandb', or 'none').",
    )
    return parser.parse_args()


def get_deepspeed_config(per_device_batch_size: int, gradient_accumulation_steps: int) -> Dict[str, Any]:
    """Construct a DeepSpeed ZeRO‑3 configuration.

    ZeRO‑3 partitions parameters, gradients and optimizer states across
    processes and offloads them to CPU.  This configuration works well
    for single‑GPU fine‑tuning of large models on 40 GB GPUs.
    """
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
        },
        "train_micro_batch_size_per_gpu": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        # enable flash attention automatically when supported
        "flops_profiler": {"enabled": False},
    }
    return config

def get_image_placeholder(processor) -> str:
    """Return the image placeholder string expected by this processor."""
    ph = getattr(processor, "image_token", None)
    if ph:
        return ph
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        ph = getattr(tok, "image_token", None)
        if ph:
            return ph
        ph = (getattr(tok, "special_tokens_map", {}) or {}).get("start_image_token")
        if ph:
            return ph
        ph = (getattr(tok, "special_tokens_map", {}) or {}).get("image_token")
        if ph:
            return ph
    # HF InternVL3-hf commonly uses <img> ... </img> with <IMG_CONTEXT>;
    # the plain "<image>" also works in many builds. We pick a safe default.
    return "<image>"

import torch

def _pick_first(x):
    # If processor returns a list/tuple (e.g., variable-size outputs), pick first
    return x[0] if isinstance(x, (list, tuple)) else x

def _to_tensor(x):
    # Convert numpy/pythonic arrays to torch tensor
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

def _rm_batch1(x):
    # pick-first -> tensor -> squeeze leading singleton batch dim (if any)
    x = _pick_first(x)
    x = _to_tensor(x)
    return x.squeeze(0) if (x.ndim >= 1 and x.shape[0] == 1) else x

def _grab_tensor(x):
    """
    Robustly convert processor output to a torch.Tensor without over-indexing.
    - If x is a list/tuple, take the first element.
    - If it's already a tensor, keep it.
    - Else convert via torch.as_tensor.
    - If it has a leading batch dim of 1, squeeze it.
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    # Squeeze only a leading singleton batch dim
    if x.ndim >= 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x

def _rm_batch(t):
        # remove a leading batch dim of size 1 if present
        return t.squeeze(0) if hasattr(t, "dim") and t.dim() > 0 and t.shape[0] == 1 else t
    
def prepare_example(
    example: Dict[str, Any],
    processor: Any,
    no_answer_template: Any,  # kept for signature compatibility; unused
    lora_image_ids: List[int],
    max_length: int,          # kept for signature compatibility; not used here to avoid truncation issues
    image_key: str = "decoded_image",
) -> Dict[str, Any]:
    """Prepare one MathVista example for InternVL3 SFT with LoRA.
    Returns: dict with input_ids, pixel_values, attention_mask, labels (all tensors).
    """
    # 1) Build question text (choices, unit, precision)
    question = example["question"]
    choices = example.get("choices")
    if choices:
        letters = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
        question += "\nChoices:\n" + "\n".join(letters)

    unit = example.get("unit")
    if unit and unit != "null":
        question += f"\nUnit: {unit}"

    precision = example.get("precision")
    if precision is not None and str(precision) != "null":
        question += f"\nPlease provide the answer with {precision} decimal places."

    answer = str(example["answer"])
    pil_img = example[image_key]  # PIL.Image or datasets.Image

    # 2) Messages WITH the image embedded (do NOT pass images= kwarg later)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": question},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]
    no_answer_messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": question},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": ""},
        ]},
    ]

    # 3) Let the processor build BOTH text and image tokens together.
    #    No truncation here to avoid slicing the image-token block.
    full_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )
    prefix_inputs = processor.apply_chat_template(
        no_answer_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )

    
    
    # get tensors as returned by the processor
    input_ids      = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    pixel_values   = full_inputs["pixel_values"]
    pref_ids       = prefix_inputs["input_ids"]
    
    # squeeze a leading batch dim IF it exists (and only if it is 1)
    if input_ids.dim() == 2 and input_ids.size(0) == 1:
        input_ids = input_ids.squeeze(0)
    if attention_mask.dim() == 2 and attention_mask.size(0) == 1:
        attention_mask = attention_mask.squeeze(0)
    # pixel_values is usually [1, C, H, W]; squeeze the batch dim if present
    if pixel_values.dim() >= 4 and pixel_values.size(0) == 1:
        pixel_values = pixel_values.squeeze(0)
    # prefix ids too
    if pref_ids.dim() == 2 and pref_ids.size(0) == 1:
        pref_ids = pref_ids.squeeze(0)
    
    # compute prefix length safely
    prefix_length = pref_ids.size(-1)


    # 5) Labels: copy input_ids, mask prefix and all known image special token ids
    labels = input_ids.clone()
    labels[:prefix_length] = -100
    for tok_id in lora_image_ids:
        labels[labels == tok_id] = -100

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }



def collate_fn(examples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Collate a batch of variable‑length multimodal examples.

    This function pads the input_ids and labels to the maximum length in
    the batch, stacks the pixel_values tensors and creates an
    attention_mask.  Tokens in the labels corresponding to padding are
    set to –100.
    """
    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    attention_masks = [ex["attention_mask"] for ex in examples]
    # Pad input_ids and labels
    max_len = max(x.size(0) for x in input_ids)
    padded_input_ids = torch.full(
        (len(input_ids), max_len), pad_token_id, dtype=input_ids[0].dtype
    )
    padded_labels = torch.full(
        (len(labels), max_len), -100, dtype=labels[0].dtype
    )
    padded_attention_mask = torch.zeros(
        (len(attention_masks), max_len), dtype=attention_masks[0].dtype
    )
    for i, (ids, lbls, attn) in enumerate(zip(input_ids, labels, attention_masks)):
        seq_len = ids.size(0)
        padded_input_ids[i, :seq_len] = ids
        padded_labels[i, :seq_len] = lbls
        padded_attention_mask[i, :seq_len] = attn
    return {
        "input_ids": padded_input_ids,
        "pixel_values": pixel_values,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Write DeepSpeed config to disk
    ds_config = get_deepspeed_config(
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    ds_config_path = os.path.join(args.output_dir, "ds_config.json")
    with open(ds_config_path, "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)
    # Load processor (image+text) and model.  The processor supplies the
    # chat template used to build the prompts.  Flash attention is
    # enabled via use_flash_attn=True【23197231296549†L303-L314】.
    # Load the processor and model.  For HF-compliant InternVL3 checkpoints (those
    # ending with "-hf"), `trust_remote_code` and `use_flash_attn` are not
    # required because the model architecture is implemented natively in
    # Transformers.  We keep `low_cpu_mem_usage=True` and `torch_dtype=bfloat16`
    # to reduce memory footprint.  If you are using a non-hf checkpoint you
    # may need to pass `trust_remote_code=True` and potentially `use_flash_attn=True`.
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Freeze the vision encoder and projector; only train the language model via LoRA
    for name, param in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            param.requires_grad_(False)
    # Identify the image token IDs used by InternVL so we can mask them
    image_ids = [
        processor.tokenizer.start_image_token_id,
        processor.tokenizer.context_image_token_id,
        processor.tokenizer.end_image_token_id,
    ]
    # Build LoRA configuration.  We only target linear projection layers inside
    # the language model.  These layer names cover attention and MLP
    # projections in Qwen2/InternLM backbones.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    # Print the number of trainable parameters for verification
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    # Load the dataset.  The testmini subset contains 1k samples【92535232541055†L71-L81】.
    # dataset = load_dataset(
    #     args.dataset_name,
    #     split=args.subset,
    #     use_auth_token=False,
    # )
    dataset = load_dataset(args.dataset_name, split=args.subset)
    cols = dataset.column_names
    image_key = "decoded_image" if "decoded_image" in cols else "image"

    # Convert the 'decoded_image' column to PIL images if present; otherwise use
    # the 'image' column (datasets.Image feature) directly.
    # if "decoded_image" in dataset.column_names:
    #     dataset = dataset.rename_column("decoded_image", "image")
    # else:
    #     # Cast the image column to the Image feature so that it yields PIL images
    #     try:
    #         from datasets import Image

    #         dataset = dataset.cast_column("image", Image())
    #     except Exception:
    #         # Fallback: images are already loaded as PIL objects
    #         pass
    
    # Preprocess the dataset.  We wrap the processor and other objects
    # in the lambda so that map can serialise them.
    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_example(
            example,
            processor=processor,
            no_answer_template=None,
            lora_image_ids=image_ids,
            max_length=args.max_length,
            image_key=image_key,
        )
    processed_dataset = dataset.map(
        _preprocess,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
    )
    
    processed_dataset = processed_dataset.with_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "pixel_values"],
    )
    
    # Data collator closure
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_fn(features, pad_token_id=pad_token_id)
    # Prepare training arguments
    total_train_steps = None  # Let Trainer infer from epoch count
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=args.report_to,
        deepspeed=ds_config_path,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    # Start training
    trainer.train()
    # Save the LoRA adapters for later use
    trainer.save_model(os.path.join(args.output_dir, "lora"))


if __name__ == "__main__":
    main()