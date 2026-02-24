#!/usr/bin/env python3
"""Evaluate PyTorch ResNet8 model on CIFAR-10 test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resnet8.evaluation import (  # noqa: E402
    DEFAULT_CIFAR10_DATA_DIR,
    PyTorchAdapter,
    evaluate_dataset,
    format_report_text,
    load_cifar10_test,
    write_report_json,
)


def _format_quant_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _print_quantization_summary(adapter: PyTorchAdapter) -> None:
    rows = adapter.describe_quantization()
    if not rows:
        return

    print("Layer quantization parameters:")
    print(
        "layer | type | tensor | bits | scheme | calibrated | "
        "scale | zero_point | qmin | qmax"
    )
    print(
        "----- | ---- | ------ | ---- | ------ | ---------- | "
        "----- | ---------- | ---- | ----"
    )
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["layer"]),
                    str(row["layer_type"]),
                    str(row["tensor"]),
                    str(row["bits"]),
                    str(row["scheme"]),
                    str(row["calibrated"]),
                    _format_quant_value(row["scale"]),
                    _format_quant_value(row["zero_point"]),
                    _format_quant_value(row["qmin"]),
                    _format_quant_value(row["qmax"]),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch ResNet8 model on CIFAR-10 test set"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.pt",
        help="Path to PyTorch model file (default: models/resnet8.pt)",
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
    parser.add_argument(
        "--wq",
        type=int,
        default=None,
        help="Optional PTQ weight bit-width (2-16)",
    )
    parser.add_argument(
        "--aq",
        type=int,
        default=None,
        help="Optional PTQ activation bit-width (2-16)",
    )
    parser.add_argument(
        "--aq-scheme",
        choices=("symmetric", "asymmetric"),
        default="symmetric",
        help="Activation quantization scheme (default: symmetric)",
    )
    parser.add_argument(
        "--calib",
        action="store_true",
        help=(
            "Use calibration-derived quantization parameters from "
            "<data-dir>/test_batch for PTQ simulation"
        ),
    )
    parser.add_argument(
        "--pre-channel",
        "--per-channel",
        dest="per_channel",
        action="store_true",
        help="Use per-channel weight quantization when --wq is set",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device: auto (default), cpu, or cuda",
    )
    args = parser.parse_args()
    if args.calib and args.wq is None and args.aq is None:
        parser.error("--calib requires --wq and/or --aq")

    print(f"Loading CIFAR-10 test data from: {args.data_dir}")
    print(f"Loading PyTorch model from: {args.model}")
    if args.wq is not None or args.aq is not None:
        print(
            "Applying PTQ simulation: "
            f"wq={args.wq}, aq={args.aq}, aq_scheme={args.aq_scheme}, "
            f"weight_mode={'per-channel' if args.per_channel else 'per-tensor'}"
        )
    if args.calib:
        print(f"Calibrating PTQ parameters from: {Path(args.data_dir) / 'test_batch'}")
    print(f"Requested device: {args.device}")

    calibration_images = None
    if args.calib:
        calibration_images, _, _ = load_cifar10_test(args.data_dir)

    try:
        adapter = PyTorchAdapter(
            args.model,
            weight_bits=args.wq,
            activation_bits=args.aq,
            activation_scheme=args.aq_scheme,
            calibrate=args.calib,
            calibration_images=calibration_images,
            device=args.device,
            per_channel=args.per_channel,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Using device: {adapter.device}")
    print("Model loaded and set to eval mode")
    _print_quantization_summary(adapter)

    report = evaluate_dataset(
        adapter=adapter,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )

    print()
    print(format_report_text(report, title="EVALUATION RESULTS (PyTorch)"))

    if args.output_json:
        write_report_json(report, args.output_json)
        print(f"Wrote JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
