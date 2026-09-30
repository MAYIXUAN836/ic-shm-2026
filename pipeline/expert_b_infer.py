#!/usr/bin/env python3
"""Run one frozen Expert B adapter on the final-approval reference set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from prompt_config import SYSTEM_PROMPT


LEGACY_RUNNER = Path(__file__).resolve().with_name("legacy_runner.py")
spec = importlib.util.spec_from_file_location("legacy_runner", LEGACY_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {LEGACY_RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def canonical_messages(image_path: Path, categories: list[str]) -> list[dict[str, Any]]:
    category_json = json.dumps(categories, ensure_ascii=False)
    user_text = (
        "Expert: expert_b\n"
        f"Frozen Stage 1 categories to describe: {category_json}\n"
        "Describe these categories only. Do not add or discuss any other category.\n"
        "Return strict JSON now."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interface", choices=("canonical", "legacy"), required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/base/Qwen3_VL_8B_Instruct",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    rows = runner.read_jsonl(args.input_jsonl)
    if not rows:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("predictions.jsonl", "diagnostics.jsonl"):
            (args.output_dir / name).write_text("")
        print("Empty Expert B route; model loading skipped.")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    processor = AutoProcessor.from_pretrained(
        str(args.adapter.resolve()),
        min_pixels=50_176,
        max_pixels=100_352,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model.resolve()),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter.resolve()))
    model.eval()

    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = []
        images = []
        try:
            for row in batch:
                categories = row["damage_categories"]
                image_path = Path(row["image_path"])
                messages = (
                    canonical_messages(image_path, categories)
                    if args.interface == "canonical"
                    else runner.make_messages("expert_b", image_path, categories)
                )
                messages[1]["content"][1]["text"] = "Q2: Describe the damage characteristics based on the image?\n" + messages[1]["content"][1]["text"]
                prompts.append(
                    processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            )
        finally:
            for image in images:
                image.close()
        inputs = {
            key: value.to("cuda:0") if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        raw_outputs = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for row, raw in zip(batch, raw_outputs, strict=True):
            categories = row["damage_categories"]
            reported_raw, description, flags = runner.parse_raw_output(raw)
            reported = reported_raw
            if reported_raw is not None and args.interface == "legacy":
                reported = [
                    runner.CANONICAL_NAMES.get(category, category)
                    for category in reported_raw
                ]
            if reported is not None and reported != categories:
                flags.append("model_category_mismatch")
            if description is not None:
                mentioned = runner.mentioned_categories(description)
                missing = sorted(set(categories) - mentioned)
                outside = sorted(mentioned - set(categories))
                if missing:
                    flags.append("supplied_category_omission:" + ",".join(missing))
                if outside:
                    flags.append("outside_category_language:" + ",".join(outside))
                if runner.FORBIDDEN.search(description):
                    flags.append("forbidden_content")
            predictions.append(
                {
                    "record_id": row["record_id"],
                    "image_id": row["image_id"],
                    "damage_categories": categories,
                    "description": description or "",
                }
            )
            diagnostics.append(
                {
                    "record_id": row["record_id"],
                    "image_id": row["image_id"],
                    "stage1_categories": categories,
                    "raw_model_output": raw,
                    "model_reported_categories": reported,
                    "model_reported_categories_raw": reported_raw,
                    "parsed_description": description,
                    "validation_flags": flags,
                }
            )
        print(f"{args.interface}: {len(predictions)}/{len(rows)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "diagnostics.jsonl", diagnostics)
    summary = {
        "records": len(predictions),
        "interface": args.interface,
        "json_errors": sum(
            any(flag.startswith("invalid_") or flag == "json_not_object" for flag in row["validation_flags"])
            for row in diagnostics
        ),
        "category_array_errors": sum(
            "model_category_mismatch" in row["validation_flags"] for row in diagnostics
        ),
        "omission_records": sum(
            any(flag.startswith("supplied_category_omission:") for flag in row["validation_flags"])
            for row in diagnostics
        ),
        "outside_category_records": sum(
            any(flag.startswith("outside_category_language:") for flag in row["validation_flags"])
            for row in diagnostics
        ),
        "forbidden_content_records": sum(
            "forbidden_content" in row["validation_flags"] for row in diagnostics
        ),
        "flagged_records": sum(bool(row["validation_flags"]) for row in diagnostics),
        "unique_descriptions": len({row["description"] for row in predictions}),
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
