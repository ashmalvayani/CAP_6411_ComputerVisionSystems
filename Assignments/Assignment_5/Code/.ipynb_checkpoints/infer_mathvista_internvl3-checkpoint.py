"""
InternVL-3 (LoRA merged) inference for MathVista.

Example:
python infer_mathvista_internvl3.py \
  --model_id OpenGVLab/InternVL3-8B \
  --lora_path InternVL/internvl_chat/work_dirs/internvl_chat_v3/internvl3_8b_dynamic_res_2nd_finetune_full \
  --split test \
  --output results/internvl3_test.json \
  --dtype bfloat16 --batch_size 1 --temperature 0.0
"""


import argparse, json, os
from typing import Optional, Tuple, List
from tqdm import tqdm
import torch
from datasets import load_dataset
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoImageProcessor,
)

try:
    from peft import PeftModel
    PEFT = True
except Exception:
    PEFT = False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--split", default="test", choices=["test", "testmini"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def to_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def task_instruction(qtype: Optional[str], atype: Optional[str], precision: Optional[str]) -> str:
    qtype = (qtype or "").lower().replace("_", "-")
    atype = (atype or "").lower()
    if qtype.startswith("multi"):
        return "Output ONLY the final option letter in uppercase (A, B, C, D, E, F, ...). No other text."
    if atype == "integer":
        return "Output ONLY a single integer. No units or extra text."
    if atype.startswith("float"):
        p = 2
        try:
            p = int(precision) if precision not in (None, "none") else 2
        except Exception:
            p = 2
        return f"Output ONLY one number rounded to exactly {p} decimal place(s). No units or extra text."
    if atype == "list":
        return "Output ONLY a Python list literal with the final values, e.g., [1, 2, 3] or [1.2, 1.3]."
    return "Output ONLY the final answer with no explanation or units."


def find_image_token(tokenizer, processor) -> Tuple[str, int]:
    # Choose an existing token (no resize of embeddings)
    candidates: List[str] = []
    if hasattr(processor, "image_token") and isinstance(processor.image_token, str):
        candidates.append(processor.image_token)
    candidates += ["<image>", "<img>", "<image_placeholder>", "<vision>"]

    for t in candidates:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid not in (-1, tokenizer.unk_token_id):
            return t, tid

    if getattr(tokenizer, "additional_special_tokens", None):
        for t in tokenizer.additional_special_tokens:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid not in (-1, tokenizer.unk_token_id):
                return t, tid

    # Last resort (not ideal semantically, but avoids crash)
    return tokenizer.eos_token, tokenizer.eos_token_id


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("loading the dataset")
    ds = load_dataset("AI4Math/MathVista")
    data = ds[args.split]
    print("Dataset loaded")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch_dtype = to_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # ---- pad_token_id setup to silence the message ----
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token       # reuse EOS as PAD for decoder-only LMs
    tokenizer.padding_side = "left"                     # safer for batched decoding
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    # ---------------------------------------------------

    print("Model Loaded")

    if args.lora_path:
        if not PEFT:
            raise RuntimeError("peft is not installed but lora_path provided.")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("lora weights added to the model")

    model.eval()

    image_token_str, img_tok_id = find_image_token(tokenizer, processor)
    setattr(model, "img_context_token_id", img_tok_id)
    print(f"[internvl] using image token: {repr(image_token_str)} (id={img_tok_id})")

    results = []
    bs = args.batch_size

    for i in tqdm(range(0, len(data), bs)):
        examples = [data[j] for j in range(i, min(i + bs, len(data)))]

        # 1) load PIL images and fields
        pil_images, pids, instrs, queries = [], [], [], []
        for ex in examples:
            img_obj = ex.get("decoded_image", None)
            if isinstance(img_obj, Image.Image):
                img = img_obj
            else:
                img_field = ex["image"]
                img_path = img_field["path"] if isinstance(img_field, dict) and "path" in img_field else img_field
                img = Image.open(img_path).convert("RGB")

            pil_images.append(img)
            pids.append(str(ex["pid"]))
            instrs.append(task_instruction(ex.get("question_type"), ex.get("answer_type"), ex.get("precision")))
            queries.append(ex["query"])

        # 2) preprocess images
        image_inputs = image_processor(images=pil_images, return_tensors="pt")
        image_inputs = {k: v.to(model.device) for k, v in image_inputs.items()}

        # 3) cast images to model dtype BEFORE feature extraction
        target_dtype = getattr(model, "dtype", torch_dtype)
        for key in ("pixel_values", "images"):
            if key in image_inputs:
                image_inputs[key] = image_inputs[key].to(dtype=target_dtype)

        # 4) get EXACT number of vision tokens from the model
        vit_embeds = model.extract_feature(image_inputs["pixel_values"])
        if vit_embeds.dim() != 3:
            raise RuntimeError(f"Unexpected vit_embeds shape: {tuple(vit_embeds.shape)}")
        per_sample_N = vit_embeds.shape[1]

        # Build prompts with exactly N placeholders
        placeholders = " ".join([image_token_str] * per_sample_N)
        texts = [f"{placeholders}\n{instrs[idx]}\n\n{queries[idx]}" for idx in range(len(examples))]

        # 5) tokenize text
        text_inputs = tokenizer(texts, padding=True, return_tensors="pt")
        text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}

        # 6) merge dicts; DO NOT pass an 'images' kwarg into generate()
        enc = {**text_inputs, **image_inputs}
        if "images" in enc:
            del enc["images"]  # LM doesn't accept this kwarg

        # 7) generate  (no use_cache here; InternVL forwards it internally)
        do_sample = args.temperature > 0
        gen = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            **({"temperature": args.temperature, "top_p": args.top_p} if do_sample else {}),
            pad_token_id=tokenizer.pad_token_id,  # ensure silence + consistency
        )
        outs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for pid, out in zip(pids, outs):
            results.append({"pid": pid, "response": out})

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved:", args.output, "items:", len(results))


if __name__ == "__main__":
    main()
