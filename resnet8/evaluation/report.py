"""Evaluation report schema, formatting, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resnet8.evaluation.metrics import ClassMetrics, OverallMetrics


def build_report(
    *,
    backend: str,
    model_path: str,
    data_dir: str,
    overall: OverallMetrics,
    per_class: list[ClassMetrics],
) -> dict[str, Any]:
    """Build canonical evaluation report dictionary."""
    return {
        "schema_version": "1.0",
        "backend": backend,
        "model_path": model_path,
        "data_dir": data_dir,
        "overall": {
            "correct": overall.correct,
            "total": overall.total,
            "accuracy": overall.accuracy,
        },
        "per_class": [
            {
                "class_name": m.class_name,
                "correct": m.correct,
                "total": m.total,
                "accuracy": m.accuracy,
            }
            for m in per_class
        ],
    }


def format_report_text(report: dict[str, Any], title: str) -> str:
    """Format evaluation report for terminal output."""
    lines = ["=" * 50, title, "=" * 50, ""]

    overall = report["overall"]
    lines.append(
        "Overall Accuracy: "
        f"{overall['correct']}/{overall['total']} = {overall['accuracy'] * 100:.2f}%"
    )
    lines.append("")

    lines.append("Per-Class Accuracy:")
    lines.append("-" * 40)
    for row in report["per_class"]:
        lines.append(
            f"  {row['class_name']:12s}: {row['correct']:4d}/{row['total']:4d} = "
            f"{row['accuracy'] * 100:5.2f}%"
        )
    lines.append("-" * 40)
    return "\n".join(lines)


def write_report_json(report: dict[str, Any], output_path: str | Path) -> None:
    """Write evaluation report to JSON file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
