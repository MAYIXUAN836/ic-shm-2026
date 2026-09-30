#!/usr/bin/env python3
"""Score descriptions with METEOR without synonym matching and token F1."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.translate.meteor_score import meteor_score


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class NoSynonymWordNet:
    @staticmethod
    def synsets(_: str) -> list[Any]:
        return []


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def meteor(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = tokens(reference), tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return float(meteor_score([reference_tokens], prediction_tokens, wordnet=NoSynonymWordNet()))


def token_f1(reference: str, prediction: str) -> float:
    ref, pred = Counter(tokens(reference)), Counter(tokens(prediction))
    overlap = sum((ref & pred).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def json_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line and line not in ("[", "]"):
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-format", choices=("jsonl", "q2-json"), required=True)
    parser.add_argument("--reference-field", default="description_label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-image-output", type=Path, required=True)
    args = parser.parse_args()

    predictions = {row["image_id"]: row for row in json_lines(args.predictions)}
    reference_rows = json_lines(args.references)
    if args.reference_format == "jsonl":
        references = {row["image_id"]: row[args.reference_field] for row in reference_rows}
    else:
        references = {
            Path(row["img"]).stem.lower(): row["label"]
            for row in reference_rows if "\nA:" in row.get("prompt", "")
        }

    rows = []
    for image_id, prediction in predictions.items():
        key = image_id if args.reference_format == "jsonl" else Path(image_id).stem.lower()
        if key not in references:
            raise ValueError(f"Missing reference for {image_id}")
        reference = references[key]
        rows.append({
            "image_id": image_id,
            "reference": reference,
            "prediction": prediction["description"],
            "meteor_no_synonym_matching": meteor(reference, prediction["description"]),
            "token_f1": token_f1(reference, prediction["description"]),
        })
    report = {
        "records": len(rows),
        "mean_meteor_no_synonym_matching": sum(row["meteor_no_synonym_matching"] for row in rows) / len(rows),
        "mean_token_f1": sum(row["token_f1"] for row in rows) / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.per_image_output.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
