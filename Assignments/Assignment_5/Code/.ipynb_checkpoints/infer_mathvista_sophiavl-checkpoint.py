#!/usr/bin/env python

"""
# SophiaVL-R1 (LoRA merged)
python infer_mathvista_sophiavl.py \
  --model_id bunny127/SophiaVL-R1-7B \
  --lora_path Qwen2-VL-Finetune/output/testing_lora \
  --split test \
  --output results/sophiavl_r1_test.json \
  --dtype bfloat16 --batch_size 1 --temperature 0.0
"""

import argparse, json, os
from typing import Dict
from tqdm import tqdm
import torch
from datasets import load_dataset
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration

try:
    from peft import PeftModel
    PEFT=True
except Exception:
    PEFT=False

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="e.g., bunny127/SophiaVL-R1-7B or your local fine-tuned base")
    ap.add_argument("--lora_path", default=None, help="Optional PEFT adapter folder for your LoRA")
    ap.add_argument("--split", default="test", choices=["test","testmini"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def to_dtype(s): return {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[s]

def task_instruction(qtype: str, atype: str, precision: str) -> str:
    """
    Build a strict instruction that mirrors MathVista's table.
    We keep this short and deterministic, to minimize off-format generations.
    """
    qtype = (qtype or "").lower().replace("_","-")
    atype = (atype or "").lower()

    if qtype.startswith("multi"):
        return ("You must output ONLY the final option letter in uppercase "
                "(A, B, C, D, E, F, ...). Do not include words, punctuation, or reasoning.")

    # free-form
    if atype == "integer":
        return ("You must output ONLY a single integer (e.g., 1). "
                "No units, no extra text, no punctuation.")

    if atype.startswith("float"):
        # precision can be "1" or "2"
        p = 2
        try:
            p = int(precision) if precision not in (None,"none") else 2
        except Exception:
            p = 2
        return (f"You must output ONLY one floating-point number rounded to exactly {p} decimal place(s). "
                "Use standard rounding. No units or extra text.")

    if atype == "list":
        return ("You must output ONLY a Python list literal with the final values, e.g., [1, 2, 3] or [1.2, 1.3]. "
                "No explanation or extra text.")

    # safe fallback
    return ("You must output ONLY the final answer with no explanation, punctuation, or units. "
            "If multiple values are required, output a Python list literal like [a, b].")

@torch.no_grad()
def main():
    args = args_parse()
    torch.manual_seed(args.seed)

    # dataset (images + query + meta fields)
    print("loading the dataset")
    ds = load_dataset("AI4Math/MathVista")
    data = ds[args.split]
    print("Dataset loaded")

    dtype = to_dtype(args.dtype)
    device_map = "auto"

    # load model/processor/tokenizer
    _ = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, device_map=device_map, attn_implementation="flash_attention_2", 
            trust_remote_code=True
        )
    
    print("Model Loaded")
    
    if args.lora_path:
        if not PEFT:
            raise RuntimeError("peft is not installed but lora_path provided.")
        model = PeftModel.from_pretrained(model, args.lora_path)
        # merged adapters reduce VRAM for inference
        model = model.merge_and_unload()

        print("lora weights added to the model")
    model.eval()

    results = []
    bs = args.batch_size
    for i in tqdm(range(0, len(data), bs)):
        # always a list of dicts
        examples = [data[j] for j in range(i, min(i + bs, len(data)))]
    
        pil_images, messages_texts, pids = [], [], []
        for ex in examples:
            # robust image extraction (handles PIL, dict, or path)
            img_obj = ex.get("decoded_image", None)
            if isinstance(img_obj, Image.Image):
                img = img_obj
            else:
                img_field = ex["image"]
                if isinstance(img_field, dict) and "path" in img_field:
                    img_path = img_field["path"]
                else:
                    img_path = img_field  # already a path string
                img = Image.open(img_path).convert("RGB")

            # --- strict task instruction per item ---
            instr = task_instruction(ex.get("question_type"),
                                     ex.get("answer_type"),
                                     ex.get("precision"))

            # --- IMPORTANT: image item with explicit "type": "image"; put image first ---
            messages = [
                {"role": "system", "content": instr},
                {"role": "user", "content": [
                    {"type": "image", "image": img},           # <= this creates image placeholders
                    {"type": "text",  "text": ex["query"]},    # dataset-provided prompt
                ]}
            ]

            # Turn messages into a prompt containing image placeholders
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    
            pil_images.append(img)
            messages_texts.append(text)
            pids.append(str(ex["pid"]))
    
        inputs = processor(text=messages_texts, images=pil_images, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.temperature > 0),
            use_cache=True,
        )
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    
        for pid, out in zip(pids, texts):
            results.append({"pid": pid, "response": out})

    

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved:", args.output, "items:", len(results))

if __name__ == "__main__":
    main()
