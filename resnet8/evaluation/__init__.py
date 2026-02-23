"""Shared evaluation utilities for ResNet8 workflows."""

from resnet8.evaluation.adapters import OnnxRuntimeAdapter, PyTorchAdapter
from resnet8.evaluation.cifar10 import DEFAULT_CIFAR10_DATA_DIR, load_cifar10_test
from resnet8.evaluation.metrics import (
    ClassMetrics,
    OverallMetrics,
    compute_overall_metrics,
    compute_per_class_metrics,
)
from resnet8.evaluation.pipeline import evaluate_arrays, evaluate_dataset
from resnet8.evaluation.report import (
    build_report,
    format_report_text,
    write_report_json,
)

__all__ = [
    "DEFAULT_CIFAR10_DATA_DIR",
    "ClassMetrics",
    "OverallMetrics",
    "OnnxRuntimeAdapter",
    "PyTorchAdapter",
    "build_report",
    "compute_overall_metrics",
    "compute_per_class_metrics",
    "evaluate_arrays",
    "evaluate_dataset",
    "format_report_text",
    "load_cifar10_test",
    "write_report_json",
]
