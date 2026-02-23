#!/usr/bin/env python3
"""Evaluate ONNX ResNet8 model on CIFAR-10 test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resnet8.evaluation import (  # noqa: E402
    DEFAULT_CIFAR10_DATA_DIR,
    OnnxRuntimeAdapter,
    evaluate_dataset,
    format_report_text,
    write_report_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX ResNet8 model on CIFAR-10 test set"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.onnx",
        help="Path to ONNX model file (default: models/resnet8.onnx)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_CIFAR10_DATA_DIR,
        help="Path to cifar-10-batches-py directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N test samples",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write canonical evaluation report JSON",
    )
    args = parser.parse_args()

    print(f"Loading CIFAR-10 test data from: {args.data_dir}")
    print(f"Loading model: {args.model}")

    adapter = OnnxRuntimeAdapter(args.model)
    print(f"Model input: {adapter.input_name}, shape: {adapter.input_shape}")

    report = evaluate_dataset(
        adapter=adapter,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )

    print()
    print(format_report_text(report, title="EVALUATION RESULTS (ONNX)"))

    if args.output_json:
        write_report_json(report, args.output_json)
        print(f"Wrote JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
