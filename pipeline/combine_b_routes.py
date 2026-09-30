#!/usr/bin/env python3
"""Recombine the two Expert B route outputs in Stage 1 order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERT_B = {"crack", "corrosion", "exposed_rebar", "efflorescence"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--crack-only", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--rebar-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    stage1 = read_jsonl(args.stage1)
    crack_rows = read_jsonl(args.crack_only)
    other_rows = read_jsonl(args.other)
    rebar_rows = read_jsonl(args.rebar_comparison) if args.rebar_comparison else []
    crack_by_id = {row["image_id"]: row for row in crack_rows}
    other_by_id = {row["image_id"]: row for row in other_rows}
    rebar_by_id = {row["image_id"]: row for row in rebar_rows}
    if len(crack_by_id) != len(crack_rows) or len(other_by_id) != len(other_rows):
        raise SystemExit("Duplicate Expert B route output")

    combined: list[dict] = []
    for row in stage1:
        categories = [
            category
            for category in row["predicted_categories"]
            if category in EXPERT_B
        ]
        if not categories:
            continue
        # v07 is retained only for the two exact single-category B cases.
        if categories == ["crack"]:
            prediction = crack_by_id.get(row["image_id"])
        elif categories == ["exposed_rebar"] and args.rebar_comparison:
            prediction = rebar_by_id.get(row["image_id"])
        else:
            prediction = other_by_id.get(row["image_id"])
        if prediction is None:
            raise SystemExit(f"Missing Expert B output: {row['image_id']}")
        if prediction["damage_categories"] != categories:
            raise SystemExit(f"Expert B category mismatch: {row['image_id']}")
        description = prediction.get("description")
        if not isinstance(description, str) or not description.strip():
            raise SystemExit(f"Empty Expert B description: {row['image_id']}")
        combined.append(
            {
                "image_id": row["image_id"],
                "damage_categories": categories,
                "description": " ".join(description.split()),
            }
        )

    expected_records = len(crack_rows) + len(other_rows)
    if len(combined) != expected_records:
        raise SystemExit("Missing Expert B route output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"expert_b_records": len(combined), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
