from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPERT_A_CATEGORIES = {
    "spalling",
    "void",
    "pothole",
    "honeycomb",
    "looseness",
}
EXPERT_B_CATEGORIES = {
    "crack",
    "corrosion",
    "exposed_rebar",
    "efflorescence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Stage 1, Expert A, and Expert B into final JSONL."
    )
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--expert-a", type=Path, required=True)
    parser.add_argument("--expert-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neutral-extent", action="store_true", help="Omit unmeasured extent qualifiers in repeated concrete-spalling templates")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage1_categories(row: dict[str, Any]) -> list[str]:
    for key in ("predicted_categories", "predicted_labels", "damage_categories"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(category) for category in value]
    raise ValueError(f"No category list found for {row.get('image_id')}")


def index_expert(
    rows: list[dict[str, Any]],
    expected_categories: set[str],
    name: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if list(row) != ["image_id", "damage_categories", "description"]:
            raise ValueError(f"{name} has an invalid schema: {row}")
        image_id = row["image_id"]
        if image_id in index:
            raise ValueError(f"{name} contains duplicate image ID {image_id}")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError(f"{name} contains an empty description for {image_id}")
        if not set(row["damage_categories"]).issubset(expected_categories):
            raise ValueError(f"{name} contains out-of-scope categories for {image_id}")
        index[image_id] = row
    return index


def join_descriptions(parts: list[str]) -> str:
    cleaned = [" ".join(part.split()).strip() for part in parts if part.strip()]
    if not cleaned:
        raise ValueError("No expert description is available.")
    unique_sentences: list[str] = []
    seen: set[str] = set()
    for part in cleaned:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", part)
            if sentence.strip()
        ]
        for sentence in sentences:
            key = re.sub(r"\s+", " ", sentence).casefold().rstrip(".!?")
            if key not in seen:
                seen.add(key)
                unique_sentences.append(sentence)
    return " ".join(
        sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        for sentence in unique_sentences
    )


def main() -> None:
    args = parse_args()
    stage1_rows = read_jsonl(args.stage1)
    expert_a = index_expert(
        read_jsonl(args.expert_a), EXPERT_A_CATEGORIES, "Expert A"
    )
    expert_b = index_expert(
        read_jsonl(args.expert_b), EXPERT_B_CATEGORIES, "Expert B"
    )

    stage1_ids: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    both_experts = 0
    for row in stage1_rows:
        image_id = row["image_id"]
        if image_id in stage1_ids:
            raise ValueError(f"Stage 1 contains duplicate image ID {image_id}")
        stage1_ids.add(image_id)
        categories = stage1_categories(row)
        expected_a = [x for x in categories if x in EXPERT_A_CATEGORIES]
        expected_b = [x for x in categories if x in EXPERT_B_CATEGORIES]
        actual_a = expert_a.get(image_id)
        actual_b = expert_b.get(image_id)
        if bool(expected_a) != bool(actual_a):
            raise ValueError(f"Expert A routing mismatch for {image_id}")
        if bool(expected_b) != bool(actual_b):
            raise ValueError(f"Expert B routing mismatch for {image_id}")
        if actual_a and actual_a["damage_categories"] != expected_a:
            raise ValueError(f"Expert A category mismatch for {image_id}")
        if actual_b and actual_b["damage_categories"] != expected_b:
            raise ValueError(f"Expert B category mismatch for {image_id}")
        descriptions = []
        if actual_a:
            descriptions.append(actual_a["description"])
        if actual_b:
            descriptions.append(actual_b["description"])
        if len(descriptions) == 2:
            both_experts += 1
        # The final category array comes directly from Stage 1; merging only
        # combines and normalizes the selected expert descriptions.
        output_rows.append(
            {
                "image_id": image_id,
                "damage_categories": categories,
                "description": join_descriptions(descriptions) if categories else "No visible structural damage is observed in the image.",
            }
        )

    if args.neutral_extent:
        for row in output_rows:
            row["description"] = row["description"].replace("Large-area concrete spalling", "Concrete spalling").replace("A large amount of concrete spalling", "Concrete spalling")

    if set(expert_a) - stage1_ids or set(expert_b) - stage1_ids:
        raise ValueError("Expert output contains an image absent from Stage 1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "stage1_records": len(stage1_rows),
                "expert_a_records": len(expert_a),
                "expert_b_records": len(expert_b),
                "both_experts": both_experts,
                "output_records": len(output_rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
