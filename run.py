#!/usr/bin/env python3
"""Run the fixed FINAL SYSTEM = final image-to-answer pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "final_config.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    # Paths in final_config.json are relative to this package root after extraction.
    for name in ("stage1", "experts"):
        config["base_models"][name] = str(
            (ROOT / config["base_models"][name]).resolve()
        )
    for name, relative_path in list(config["adapters"].items()):
        config["adapters"][name] = str((ROOT / relative_path).resolve())
    return config


def collect_inventory(input_dir: Path) -> list[dict[str, str]]:
    image_paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_dir}")
    rows = []
    for path in image_paths:
        relative = path.relative_to(input_dir).as_posix()
        rows.append({"image_id": relative, "image_path": str(path.resolve())})
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Input image IDs are not unique.")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_step(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_model_files(config: dict[str, Any]) -> None:
    for model_path in config["base_models"].values():
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(f"Missing packaged base model: {model_path}")
    for adapter_path in config["adapters"].values():
        if not (Path(adapter_path) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing packaged adapter: {adapter_path}")


def validate_final_outputs(
    inventory: list[dict[str, Any]],
    stage1_path: Path,
    stage1_raw_path: Path,
    expert_a_path: Path,
    expert_b_path: Path,
    final_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = [row["image_id"] for row in inventory]
    actual_ids = [row.get("image_id") for row in final_rows]
    if actual_ids != expected_ids:
        raise RuntimeError("Final output IDs/count/order do not match the input inventory.")
    classes = config["categories"]
    for row in final_rows:
        if list(row) != ["image_id", "damage_categories", "description"]:
            raise RuntimeError(f"Final output schema mismatch: {row}")
        categories = row["damage_categories"]
        if (
            not isinstance(categories, list)
            or len(categories) != len(set(categories))
            or not all(isinstance(category, str) and category in classes for category in categories)
        ):
            raise RuntimeError(f"Invalid final category array: {row['image_id']}")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise RuntimeError(f"Empty final description: {row['image_id']}")
        if not categories and row["description"] != config["merge"]["normal_description"]:
            raise RuntimeError(f"Normal description mismatch: {row['image_id']}")

    stage1_rows = read_jsonl(stage1_path)
    raw_rows = read_jsonl(stage1_raw_path)
    if [row.get("image_id") for row in stage1_rows] != expected_ids:
        raise RuntimeError("Stage 1 output does not match the input inventory.")
    if len(raw_rows) != len(inventory) or not all(row.get("json_parse_ok") for row in raw_rows):
        raise RuntimeError("Stage 1 has a missing or invalid JSON parse result.")
    expert_a_rows = read_jsonl(expert_a_path)
    expert_b_rows = read_jsonl(expert_b_path)
    stage1_categories = {
        row["image_id"]: row["predicted_categories"] for row in stage1_rows
    }
    normal_ids = {
        image_id for image_id, categories in stage1_categories.items() if not categories
    }
    routed_ids = {row["image_id"] for row in expert_a_rows + expert_b_rows}
    if normal_ids & routed_ids:
        raise RuntimeError("A no-damage image was routed to an expert.")
    return {
        "input_records": len(inventory),
        "final_records": len(final_rows),
        "stage1_json_parse_failures": sum(not row.get("json_parse_ok") for row in raw_rows),
        "no_damage_records": len(normal_ids),
        "expert_a_records": len(expert_a_rows),
        "expert_b_records": len(expert_b_rows),
        "no_damage_expert_skips_verified": True,
        "final_schema_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    validate_model_files(config)
    inventory = collect_inventory(input_dir)
    intermediate = output_dir / "intermediate"
    inventory_path = intermediate / "final_input_inventory.jsonl"
    write_jsonl(inventory_path, inventory)

    stage1_dir = intermediate / "stage1"
    run_step([
        sys.executable,
        str(ROOT / "pipeline" / "stage1_infer.py"),
        "--inventory-jsonl", str(inventory_path),
        "--output-dir", str(stage1_dir),
        "--base-model", config["base_models"]["stage1"],
        "--adapter", config["adapters"]["stage1"],
        "--gpu", args.gpu,
        "--max-image-side", str(config["generation"]["stage1_max_image_side"]),
        "--max-new-tokens", str(config["generation"]["stage1_max_new_tokens"]),
    ])
    stage1_path = stage1_dir / "final_stage1_predictions.jsonl"
    stage1_raw_path = stage1_dir / "final_stage1_raw_generation_audit.jsonl"

    expert_a_dir = intermediate / "expert_a"
    expert_a_path = expert_a_dir / "final_expert_a_predictions.jsonl"
    run_step([
        sys.executable,
        str(ROOT / "pipeline" / "expert_a_infer.py"),
        "--expert", "expert_a",
        "--input-jsonl", str(stage1_path),
        "--adapter", config["adapters"]["expert_a"],
        "--diagnostics", str(expert_a_dir / "final_expert_a_diagnostics.jsonl"),
        "--predictions", str(expert_a_path),
        "--base-model", config["base_models"]["experts"],
        "--gpu", args.gpu,
        "--batch-size", str(config["generation"]["batch_size"]),
        "--max-new-tokens", str(config["generation"]["expert_max_new_tokens"]),
    ])

    routes_dir = intermediate / "routes"
    run_step([
        sys.executable,
        str(ROOT / "pipeline" / "prepare_routes.py"),
        "--stage1", str(stage1_path),
        "--output-dir", str(routes_dir),
    ])

    expert_b_dir = intermediate / "expert_b"
    route_runs = [
        ("expert_b_crack_only", "expert_b_crack_only.jsonl", "expert_b_v07", "legacy"),
        ("expert_b_rebar_only", "expert_b_with_rebar.jsonl", "expert_b_v07", "legacy"),
        ("expert_b_other", "expert_b_other.jsonl", "expert_b_v08", "canonical"),
    ]
    for name, input_name, adapter_name, interface in route_runs:
        # The rebar comparison run covers several B subsets; combine_b_routes.py
        # keeps its v07 output only for the exact exposed_rebar-only subset.
        run_step([
            sys.executable,
            str(ROOT / "pipeline" / "expert_b_infer.py"),
            "--input-jsonl", str(routes_dir / input_name),
            "--adapter", config["adapters"][adapter_name],
            "--output-dir", str(expert_b_dir / name),
            "--interface", interface,
            "--base-model", config["base_models"]["experts"],
            "--gpu", args.gpu,
            "--batch-size", str(config["generation"]["batch_size"]),
            "--max-new-tokens", str(config["generation"]["expert_max_new_tokens"]),
        ])

    selected_b_path = intermediate / "final_expert_b_predictions.jsonl"
    run_step([
        sys.executable,
        str(ROOT / "pipeline" / "combine_b_routes.py"),
        "--stage1", str(stage1_path),
        "--crack-only", str(expert_b_dir / "expert_b_crack_only" / "predictions.jsonl"),
        "--other", str(expert_b_dir / "expert_b_other" / "predictions.jsonl"),
        "--rebar-comparison", str(expert_b_dir / "expert_b_rebar_only" / "predictions.jsonl"),
        "--output", str(selected_b_path),
    ])

    merged_path = intermediate / "final_merged_predictions.jsonl"
    merge_command = [
        sys.executable,
        str(ROOT / "pipeline" / "merge.py"),
        "--stage1", str(stage1_path),
        "--expert-a", str(expert_a_path),
        "--expert-b", str(selected_b_path),
        "--output", str(merged_path),
    ]
    if config["merge"]["neutral_extent"]:
        merge_command.append("--neutral-extent")
    run_step(merge_command)

    final_rows = read_jsonl(merged_path)
    validation = validate_final_outputs(
        inventory, stage1_path, stage1_raw_path, expert_a_path, selected_b_path,
        final_rows, config,
    )
    final_jsonl = output_dir / "final_predictions.jsonl"
    final_json = output_dir / "final_predictions.json"
    final_csv = output_dir / "final_predictions.csv"
    shutil.copyfile(merged_path, final_jsonl)
    final_json.write_text(
        json.dumps(final_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with final_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "damage_categories", "description"])
        writer.writeheader()
        for row in final_rows:
            writer.writerow({
                "image_id": row["image_id"],
                "damage_categories": ";".join(row["damage_categories"]),
                "description": row["description"],
            })

    decision = {
        "FINAL SYSTEM": "final",
        "system_name": "final",
        "status": "complete",
        "components": config["source_provenance"],
        "stage1_adapter": config["adapters"]["stage1"],
        "expert_a_adapter": config["adapters"]["expert_a"],
        "expert_b_v07_adapter": config["adapters"]["expert_b_v07"],
        "expert_b_v08_adapter": config["adapters"]["expert_b_v08"],
        "official_test_is_not_validation": True,
        "weight_fusion": False,
        "validation": validation,
        "submission_files": [
            "final_predictions.json",
            "final_predictions.jsonl",
            "final_predictions.csv",
        ],
    }
    (output_dir / "final_run_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
