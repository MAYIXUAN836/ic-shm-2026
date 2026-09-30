#!/usr/bin/env python3
"""Run one frozen Expert on Holdout and preserve unmodified model generation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


EXPERTS = {
    "expert_a": {
        "categories": {"spalling", "void", "pothole", "honeycomb", "looseness"},
        "system_prompt": """You are Expert A in the IC-SHM 2026 structural-damage system.
Stage 1 has already classified the image. The supplied category list is frozen
and authoritative. Describe only the visible appearance, morphology, location,
or distribution of the supplied categories. Do not reclassify the image. Do
not mention, add, or assess any damage category absent from the supplied list.
Do not infer causes, measurements, severity, structural capacity, safety,
hidden damage, progression, or repair advice. Avoid causal phrases such as
\"due to\", \"caused by\", and \"triggered by\". Return exactly one JSON object with
keys damage_categories and description. Do not use Markdown.""",
    },
    "expert_b": {
        "categories": {"crack", "corrosion", "exposed_rebar", "efflorescence"},
        "system_prompt": """You are Expert 2 in the IC-SHM 2026 structural-damage system.
Stage 1 has already classified the image. The supplied category list is frozen
and authoritative. Describe only the visible appearance, morphology, location,
or distribution of the supplied categories. Do not reclassify the image. Do
not mention, add, or assess any damage category absent from the supplied list.
Do not infer causes, measurements, severity, structural capacity, safety,
hidden damage, progression, or repair advice. Avoid causal phrases such as
\"due to\", \"caused by\", and \"triggered by\". Return exactly one JSON object with
keys damage_categories and description. Do not use Markdown.""",
    },
}
MODEL_NAMES = {
    "crack": "concrete_crack",
    "corrosion": "corrosion",
    "exposed_rebar": "spalling_exposed_rebar",
    "efflorescence": "efflorescence",
}
CANONICAL_NAMES = {value: key for key, value in MODEL_NAMES.items()}
TERMS = {
    "spalling": (r"\bspall(?:ing|ed|s)?\b", r"\bmaterial loss\b"),
    "void": (r"\bvoids?\b", r"\bcavit(?:y|ies)\b", r"\bhollows?\b"),
    "pothole": (r"\bpotholes?\b", r"\broad depression\b"),
    "honeycomb": (r"\bhoneycombs?\b", r"\bporous\b"),
    "looseness": (r"\bloose(?:ness)?\b", r"\bdisplaced component\b"),
    "crack": (r"\bcrack(?:ed|ing|s)?\b", r"\bfissures?\b"),
    "corrosion": (r"\bcorrosion\b", r"\bcorroded\b", r"\brust(?:ed|ing)?\b"),
    "exposed_rebar": (r"\brebars?\b", r"\breinforcing bars?\b", r"\bexposed reinforcement\b"),
    "efflorescence": (r"\befflorescence\b", r"\bcrystalline deposit\b", r"\bpowdery deposit\b"),
}
FORBIDDEN = re.compile(
    r"\b(due to|caused by|causing|triggered|leading to|resulting from|severe|"
    r"extensive|load[- ]bearing|structural (?:support|stability|capacity|consequence)|"
    r"safety|pressure|repair|maintenance)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage1_categories(row: dict[str, Any]) -> list[str]:
    for key in ("predicted_categories", "predicted_labels", "damage_categories"):
        if isinstance(row.get(key), list):
            return [str(category) for category in row[key]]
    raise ValueError(f"No category list found for {row.get('image_id')}")


def resolve_image(row: dict[str, Any]) -> Path:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and Path(image_path).is_file():
        return Path(image_path)
    raise FileNotFoundError(f"Image not found for {row.get('image_id')}: {image_path}")


def parse_raw_output(raw: str) -> tuple[list[str] | None, str | None, list[str]]:
    flags: list[str] = []
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None, None, ["invalid_json"]
    if not isinstance(payload, dict):
        return None, None, ["json_not_object"]
    reported = payload.get("damage_categories")
    description = payload.get("description")
    if not isinstance(reported, list) or not all(isinstance(item, str) for item in reported):
        flags.append("invalid_model_categories")
        reported = None
    if not isinstance(description, str):
        flags.append("invalid_description_type")
        description = None
    elif not description.strip():
        flags.append("empty_description")
    return reported, description.strip() if isinstance(description, str) else None, flags


def mentioned_categories(description: str) -> set[str]:
    return {
        category
        for category, patterns in TERMS.items()
        if any(re.search(pattern, description, re.IGNORECASE) for pattern in patterns)
    }


def make_messages(expert: str, image_path: Path, categories: list[str]) -> list[dict[str, Any]]:
    if expert == "expert_b":
        model_categories = [MODEL_NAMES[category] for category in categories]
        expert_name = "expert_2"
    else:
        model_categories = categories
        expert_name = "expert_a"
    user_text = (
        f"Expert: {expert_name}\n"
        f"Frozen Stage 1 categories to describe: {json.dumps(model_categories)}\n"
        "Describe these categories only. Do not add or discuss any other category.\n"
        "Stage 1 regions: unavailable during initial SFT\n"
        "Return strict JSON now."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": EXPERTS[expert]["system_prompt"]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=sorted(EXPERTS), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/base/Qwen3_VL_8B_Instruct",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = read_jsonl(args.input_jsonl)
    owned = EXPERTS[args.expert]["categories"]
    routed: list[tuple[dict[str, Any], list[str], Path]] = []
    for row in rows:
        categories = [category for category in stage1_categories(row) if category in owned]
        if categories:
            routed.append((row, categories, resolve_image(row)))

    if not routed:
        for path in (args.diagnostics, args.predictions):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        print("No records routed to this expert; model loading skipped.")
        return

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

    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.time()

    for offset in range(0, len(routed), args.batch_size):
        batch = routed[offset : offset + args.batch_size]
        prompts = [
            processor.apply_chat_template(
                make_messages(args.expert, image_path, categories),
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, categories, image_path in batch
        ]
        images = []
        try:
            for _, _, image_path in batch:
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
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
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_outputs = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for (row, categories, _), raw in zip(batch, raw_outputs, strict=True):
            reported_raw, description, flags = parse_raw_output(raw)
            reported = None
            if reported_raw is not None:
                reported = [CANONICAL_NAMES.get(category, category) for category in reported_raw]
                if reported != categories:
                    flags.append("model_category_mismatch")
            if description is not None:
                outside = sorted(mentioned_categories(description) - set(categories))
                if outside:
                    flags.append("outside_category_language:" + ",".join(outside))
                if FORBIDDEN.search(description):
                    flags.append("forbidden_content")
            diagnostics.append(
                {
                    "image_id": row["image_id"],
                    "stage1_categories": categories,
                    "raw_model_output": raw,
                    "model_reported_categories": reported,
                    "model_reported_categories_raw": reported_raw,
                    "parsed_description": description,
                    "validation_flags": flags,
                }
            )
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "damage_categories": categories,
                    "description": description or "",
                }
            )
        print(f"{args.expert} G1: {len(diagnostics)}/{len(routed)}", flush=True)

    with args.diagnostics.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.predictions.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "expert": args.expert,
                "records": len(diagnostics),
                "flagged_records": sum(bool(row["validation_flags"]) for row in diagnostics),
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
