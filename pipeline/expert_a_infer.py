#!/usr/bin/env python3
"""Run frozen Expert inference with strict category copying and category scope."""

from __future__ import annotations

import importlib.util
from pathlib import Path


G1_SCRIPT = Path(__file__).resolve().with_name("legacy_runner.py")
spec = importlib.util.spec_from_file_location("g1_runner", G1_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {G1_SCRIPT}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

original_make_messages = runner.make_messages


def make_strict_messages(expert: str, image_path: Path, categories: list[str]):
    messages = original_make_messages(expert, image_path, categories)
    messages[1]["content"][1]["text"] = "Q2: Describe the damage characteristics based on the image?\n" + messages[1]["content"][1]["text"]
    constraints = (
        "\nCopy the supplied category strings exactly into damage_categories."
        " In description, explicitly describe every supplied category."
        " The following word restrictions apply only to description:"
        ' do not use "spalling", "spalled", "extensive", or "severe".'
    )
    if "crack" not in categories:
        constraints += (
            ' Do not use "crack", "cracked", "cracking", or "fissure".'
            " Even if such a feature appears visible, ignore it because it is outside"
            " the supplied categories."
        )
    messages[1]["content"][1]["text"] += constraints
    return messages


runner.make_messages = make_strict_messages


if __name__ == "__main__":
    runner.main()
