#!/usr/bin/env python3
"""Run the fixed UNIFORM_V1 Stage 1 model."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


CLASSES = [
    "crack", "void", "spalling", "corrosion", "exposed_rebar",
    "honeycomb", "looseness", "pothole", "efflorescence",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_output(text: str) -> tuple[bool, list[str], str | None]:
    match = re.search(r"\{.*\}", text.strip(), flags=re.S)
    try:
        payload = json.loads(match.group(0) if match else text)
    except (json.JSONDecodeError, AttributeError):
        return False, [], "invalid_json"
    if not isinstance(payload, dict):
        return False, [], "json_not_object"
    if "damage_categories" not in payload:
        return False, [], "missing_damage_categories"
    values = payload["damage_categories"]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return False, [], "damage_categories_not_string_list"
    normalized = {
        str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for value in values
    }
    unknown = sorted(normalized - set(CLASSES))
    if unknown:
        return False, [], "unknown_categories:" + ",".join(unknown)
    return True, [category for category in CLASSES if category in normalized], None


def prompt(image_id: str) -> str:
    definitions = {
        "crack": "thin linear fracture or fissure in concrete, steel, or coating",
        "void": "visible hollow, cavity, or missing interior material",
        "spalling": "broken or detached concrete surface, often with rough loss of material",
        "corrosion": "rusting or corrosion staining on steel or reinforcement",
        "exposed_rebar": "reinforcing steel bars visibly exposed from concrete",
        "honeycomb": "clustered porous voids or coarse aggregate exposed from poor concrete compaction",
        "looseness": "loose, displaced, or unsecured structural component or connection",
        "pothole": "localized bowl-shaped hole or depression on a pavement surface",
        "efflorescence": "white crystalline or powdery salt deposit on concrete",
    }
    labels = ", ".join(CLASSES)
    return "\n".join([
        "Q1: Determine whether there is structural damage in the image?",
        "Classify only visible structural damage in this image.",
        "Allowed labels and visual definitions:",
        *[f"- {name}: {definitions[name]}" for name in CLASSES],
        (
            f'Return exactly one JSON object: {{"image_id": "{image_id}", '
            f'"damage_categories": [labels]}}. Labels must be selected only from: '
            f"{labels}. Return [] if no allowed damage is visible. Do not add an explanation."
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor

    rows = read_jsonl(args.inventory_jsonl)
    if not rows or len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Expected a nonempty inventory with unique image IDs.")
    if args.limit is not None:
        rows = rows[:args.limit]
    if any(not Path(row["image_path"]).is_file() for row in rows):
        raise FileNotFoundError("The production inventory contains a missing image.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_name = "final_stage1_predictions.jsonl" if args.limit is None else f"final_stage1_predictions_smoke_{len(rows)}.jsonl"
    final_path = args.output_dir / final_name
    raw_path = args.output_dir / "final_stage1_raw_generation_audit.jsonl"
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    raw_partial_path = raw_path.with_suffix(raw_path.suffix + ".partial")
    if not args.resume and (partial_path.exists() or raw_partial_path.exists()):
        raise FileExistsError("Partial Stage 1 files already exist; use --resume or choose a new output directory.")
    completed = read_jsonl(partial_path) if partial_path.exists() else []
    raw_completed = read_jsonl(raw_partial_path) if raw_partial_path.exists() else []
    if len(completed) != len(raw_completed):
        raise RuntimeError("Stage 1 and raw partial files have different lengths.")
    if [row["image_id"] for row in completed] != [row["image_id"] for row in rows[:len(completed)]]:
        raise RuntimeError("Stage 1 partial output does not match the inventory prefix.")

    model_errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_class = getattr(transformers, name, None)
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(
                str(args.base_model), trust_remote_code=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                device_map={"": 0}, local_files_only=True,
            )
            break
        except Exception as error:
            model_errors.append(f"{name}: {type(error).__name__}: {error}")
    else:
        raise RuntimeError("Unable to load the local QwenV3 base model: " + " | ".join(model_errors))

    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.base_model), trust_remote_code=True, local_files_only=True
    )
    device = next(model.parameters()).device
    started = time.time()
    mode = "a" if completed else "w"
    with partial_path.open(mode, encoding="utf-8") as stage1_handle, raw_partial_path.open(mode, encoding="utf-8") as raw_handle:
        for index, row in enumerate(rows[len(completed):], len(completed) + 1):
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
                if args.max_image_side and max(image.size) > args.max_image_side:
                    image.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)
                message = {"role": "user", "content": [
                    {"type": "image", "image": row["image_path"]},
                    {"type": "text", "text": prompt(row["image_id"])},
                ]}
                text = processor.apply_chat_template(
                    [message], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {
                key: value.to(device=device, dtype=torch.bfloat16) if value.is_floating_point() else value.to(device)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
            raw_output = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            parse_ok, predicted, parse_error = parse_output(raw_output)
            stage1_handle.write(json.dumps({
                "image_id": row["image_id"], "image_path": row["image_path"],
                "classes_version": "classes_v1", "category_names": CLASSES,
                "model_name": "Q1_qwen_dacl30_v3", "model_type": "qwen3.5_9b_lora",
                "base_model": str(args.base_model), "prediction_type": "generated_label_list",
                "predicted_categories": predicted, "class_probabilities": {},
                "probability_available": False, "localization": {"available": False, "regions": []},
            }, ensure_ascii=False) + "\n")
            raw_handle.write(json.dumps({
                "index": index, "image_id": row["image_id"], "raw_output": raw_output,
                "json_parse_ok": parse_ok, "parse_error": parse_error,
                "predicted_categories": predicted,
            }, ensure_ascii=False) + "\n")
            stage1_handle.flush()
            raw_handle.flush()
            print(f"Qwen V3: {index}/{len(rows)} -> {predicted}", flush=True)

    records = read_jsonl(partial_path)
    raw_records = read_jsonl(raw_partial_path)
    if len(records) != len(rows) or len(raw_records) != len(rows):
        raise RuntimeError("Stage 1 output count check failed.")
    if not all(row["json_parse_ok"] for row in raw_records):
        raise RuntimeError("At least one QwenV3 generation failed JSON parsing.")
    partial_path.replace(final_path)
    raw_partial_path.replace(raw_path)
    summary = {"status": "complete", "records": len(records), "runtime_seconds": time.time() - started, "output": str(final_path)}
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
