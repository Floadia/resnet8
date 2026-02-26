#!/usr/bin/env python3
"""Evaluate PyTorch ResNet8 model on CIFAR-10 test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resnet8.evaluation import (  # noqa: E402
    DEFAULT_CIFAR10_DATA_DIR,
    OnnxRuntimeAdapter,
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


def _compute_unit_batch_accuracy(
    adapter: PyTorchAdapter | OnnxRuntimeAdapter,
    *,
    data_dir: str,
    max_samples: int | None,
) -> float:
    images, labels, _ = load_cifar10_test(data_dir, max_samples=max_samples)
    correct = 0
    total = int(labels.shape[0])
    for index in range(total):
        logits = adapter.predict_logits(images[index : index + 1])
        prediction = int(np.argmax(logits, axis=1)[0])
        if prediction == int(labels[index]):
            correct += 1
    return (correct / total) if total > 0 else 0.0


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
        "--dfq-bias-corr",
        action="store_true",
        help=(
            "Apply empirical bias correction from Data-Free Quantization "
            "(arXiv:1906.04721) after weight PTQ"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device: auto (default), cpu, or cuda",
    )
    parser.add_argument(
        "--export-onnx",
        default=None,
        help="Optional path to export eval-time graph as ONNX",
    )
    parser.add_argument(
        "--verify-exported-onnx",
        action="store_true",
        help=(
            "After export, evaluate the ONNX file and compare accuracy with "
            "PyTorch evaluation"
        ),
    )
    parser.add_argument(
        "--onnx-score-tol",
        type=float,
        default=0.02,
        help="Allowed absolute accuracy delta for exported ONNX parity check",
    )
    parser.add_argument(
        "--export-opset",
        type=int,
        default=17,
        help="ONNX opset version used for --export-onnx (default: 17)",
    )
    parser.add_argument(
        "--export-simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Try to simplify exported ONNX graph with onnxsim "
            "(default: enabled)"
        ),
    )
    args = parser.parse_args()
    if args.calib and args.wq is None and args.aq is None:
        parser.error("--calib requires --wq and/or --aq")
    if args.dfq_bias_corr and args.wq is None:
        parser.error("--dfq-bias-corr requires --wq")
    if args.verify_exported_onnx and args.export_onnx is None:
        parser.error("--verify-exported-onnx requires --export-onnx")
    if args.onnx_score_tol < 0:
        parser.error("--onnx-score-tol must be >= 0")

    print(f"Loading CIFAR-10 test data from: {args.data_dir}")
    print(f"Loading PyTorch model from: {args.model}")
    if args.wq is not None or args.aq is not None:
        print(
            "Applying PTQ simulation: "
            f"wq={args.wq}, aq={args.aq}, aq_scheme={args.aq_scheme}, "
            f"weight_mode={'per-channel' if args.per_channel else 'per-tensor'}, "
            f"dfq_bias_corr={args.dfq_bias_corr}"
        )
    if args.calib or args.dfq_bias_corr:
        calibration_path = Path(args.data_dir) / "test_batch"
        print(f"Loading PTQ calibration data from: {calibration_path}")
    print(f"Requested device: {args.device}")

    calibration_images = None
    if args.calib or args.dfq_bias_corr:
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
            weight_bias_correction=args.dfq_bias_corr,
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

    if args.export_onnx:
        export_path = adapter.export_onnx(
            args.export_onnx,
            opset_version=args.export_opset,
            dynamic_batch=False,
            simplify=args.export_simplify,
        )
        print(f"Exported eval graph ONNX: {export_path}")

        if args.verify_exported_onnx:
            onnx_adapter = OnnxRuntimeAdapter(export_path)
            input_shape = onnx_adapter.input_shape
            static_batch_one = bool(
                input_shape
                and isinstance(input_shape[0], int)
                and input_shape[0] == 1
            )
            if static_batch_one:
                pytorch_accuracy = _compute_unit_batch_accuracy(
                    adapter,
                    data_dir=args.data_dir,
                    max_samples=args.max_samples,
                )
                onnx_accuracy = _compute_unit_batch_accuracy(
                    onnx_adapter,
                    data_dir=args.data_dir,
                    max_samples=args.max_samples,
                )
            else:
                onnx_report = evaluate_dataset(
                    adapter=onnx_adapter,
                    data_dir=args.data_dir,
                    max_samples=args.max_samples,
                )
                pytorch_accuracy = float(report["overall"]["accuracy"])
                onnx_accuracy = float(onnx_report["overall"]["accuracy"])
            delta = abs(onnx_accuracy - pytorch_accuracy)
            print(
                "Exported ONNX parity: "
                f"pytorch={pytorch_accuracy:.6f}, "
                f"onnx={onnx_accuracy:.6f}, "
                f"abs_delta={delta:.6f}, "
                f"tol={args.onnx_score_tol:.6f}"
            )
            if delta > args.onnx_score_tol:
                raise SystemExit(
                    "Exported ONNX parity check failed: "
                    f"abs_delta={delta:.6f} exceeds tol={args.onnx_score_tol:.6f}"
                )


if __name__ == "__main__":
    main()
