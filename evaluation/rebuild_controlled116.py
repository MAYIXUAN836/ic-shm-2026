#!/usr/bin/env python3
"""Rebuild the controlled 116-image Expert A/B descriptions from frozen routes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPERT_A = {"spalling", "void", "pothole", "honeycomb", "looseness"}
EXPERT_B = {"crack", "corrosion", "exposed_rebar", "efflorescence"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {row["image_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate image IDs in {name}")
    return result


def join(parts: list[str]) -> str:
    seen, sentences = set(), []
    for part in parts:
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(part.split()).strip()):
            key = sentence.casefold().rstrip(".!?")
            if key and key not in seen:
                seen.add(key)
                sentences.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
    return " ".join(sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--expert-a", type=Path, required=True)
    parser.add_argument("--expert-b-v07", type=Path, required=True)
    parser.add_argument("--expert-b-v08", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    holdout = read_jsonl(args.holdout)
    expert_a = index(read_jsonl(args.expert_a), "Expert A")
    expert_b_v07 = index(read_jsonl(args.expert_b_v07), "Expert B v07")
    expert_b_v08 = index(read_jsonl(args.expert_b_v08), "Expert B v08")
    output = []
    for row in holdout:
        image_id, categories = row["image_id"], row["damage_categories"]
        parts, routes = [], []
        if EXPERT_A.intersection(categories):
            parts.append(expert_a[image_id]["description"])
            routes.append("expert_a")
        b_categories = [category for category in categories if category in EXPERT_B]
        if b_categories:
            use_v07 = b_categories in (["crack"], ["exposed_rebar"])
            selected = expert_b_v07 if use_v07 else expert_b_v08
            parts.append(selected[image_id]["description"])
            routes.append("expert_b_v07" if use_v07 else "expert_b_v08")
        if not parts:
            raise ValueError(f"No expert route for {image_id}")
        output.append({"image_id": image_id, "damage_categories": categories, "description": join(parts), "routes": routes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(output), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
