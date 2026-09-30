#!/usr/bin/env python3
"""Prepare the two Expert B route inputs for the v0.8 competition run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERT_A = {"spalling", "void", "pothole", "honeycomb", "looseness"}
EXPERT_B = {"crack", "corrosion", "exposed_rebar", "efflorescence"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    stage1 = read_jsonl(args.stage1)
    crack_only: list[dict] = []
    other_b: list[dict] = []
    with_rebar: list[dict] = []
    v08_non_rebar: list[dict] = []
    image_ids: set[str] = set()
    a_records = 0
    both = 0
    for index, row in enumerate(stage1, 1):
        image_id = row["image_id"]
        if image_id in image_ids:
            raise SystemExit(f"Duplicate image_id: {image_id}")
        image_ids.add(image_id)
        categories = row.get("predicted_categories")
        if not isinstance(categories, list):
            raise SystemExit(f"Missing predicted_categories: {image_id}")
        categories = [str(category) for category in categories]
        a_categories = [category for category in categories if category in EXPERT_A]
        b_categories = [category for category in categories if category in EXPERT_B]
        a_records += bool(a_categories)
        both += bool(a_categories and b_categories)
        if not categories:
            continue
        if not a_categories and not b_categories:
            raise SystemExit(f"No routed category: {image_id}")
        if b_categories:
            # Preserve B's exact subset; the final adapter choice is made on
            # this subset, not on all categories predicted for the image.
            route_row = {
                "record_id": f"competition_b_{index:04d}",
                "image_id": image_id,
                "image_path": row["image_path"],
                "damage_categories": b_categories,
            }
            if b_categories == ["crack"]:
                crack_only.append(route_row)
            else:
                other_b.append(route_row)
                if "exposed_rebar" in b_categories:
                    with_rebar.append(route_row)
                else:
                    v08_non_rebar.append(route_row)

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "expert_b_crack_only.jsonl", crack_only)
    write_jsonl(args.output_dir / "expert_b_other.jsonl", other_b)
    write_jsonl(args.output_dir / "expert_b_with_rebar.jsonl", with_rebar)
    write_jsonl(args.output_dir / "expert_b_v08_non_rebar.jsonl", v08_non_rebar)
    summary = {
        "stage1_records": len(stage1),
        "expert_a_records": a_records,
        "expert_b_records": len(crack_only) + len(other_b),
        "both_experts": both,
        "expert_b_crack_only_v07": len(crack_only),
        "expert_b_other_v08": len(other_b),
        "expert_b_with_rebar_comparison": len(with_rebar),
        "expert_b_v08_non_rebar": len(v08_non_rebar),
    }
    (args.output_dir / "route_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
