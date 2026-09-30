#!/usr/bin/env python3
"""Score multi-label category predictions against a JSONL reference file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LABELS = [
    "crack", "void", "spalling", "corrosion", "exposed_rebar",
    "honeycomb", "looseness", "pothole", "efflorescence",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested_value(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for key in field.split("."):
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-field", default="target_categories")
    parser.add_argument("--prediction-field", default="predicted_categories")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = {row["image_id"]: row for row in read_jsonl(args.predictions)}
    references = {row["image_id"]: row for row in read_jsonl(args.references)}
    if set(predictions) != set(references):
        raise ValueError("Prediction and reference image-ID sets do not match.")

    per_label = {label: Counter() for label in args.labels}
    exact = tp = fp = fn = 0
    for image_id, reference in references.items():
        target = set(nested_value(reference, args.reference_field))
        predicted = set(predictions[image_id][args.prediction_field])
        if target == predicted:
            exact += 1
        tp += len(target & predicted)
        fp += len(predicted - target)
        fn += len(target - predicted)
        for label in args.labels:
            if label in target and label in predicted:
                per_label[label]["tp"] += 1
            elif label in predicted:
                per_label[label]["fp"] += 1
            elif label in target:
                per_label[label]["fn"] += 1

    def f1(a: int, b: int, c: int) -> float:
        return 0.0 if 2 * a + b + c == 0 else 2 * a / (2 * a + b + c)

    category_metrics = {}
    for label, values in per_label.items():
        a, b, c = values["tp"], values["fp"], values["fn"]
        category_metrics[label] = {
            "support": a + c,
            "precision": 0.0 if a + b == 0 else a / (a + b),
            "recall": 0.0 if a + c == 0 else a / (a + c),
            "f1": f1(a, b, c),
        }
    report = {
        "records": len(references),
        "exact_set_match": exact,
        "exact_set_accuracy": exact / len(references),
        "micro_precision": tp / (tp + fp),
        "micro_recall": tp / (tp + fn),
        "micro_f1": f1(tp, fp, fn),
        "macro_f1": sum(row["f1"] for row in category_metrics.values()) / len(category_metrics),
        "per_category": category_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
