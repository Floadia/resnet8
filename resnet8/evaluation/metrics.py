"""Metrics for ResNet8 evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OverallMetrics:
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True)
class ClassMetrics:
    class_name: str
    correct: int
    total: int
    accuracy: float


def compute_overall_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> OverallMetrics:
    """Compute overall accuracy metrics."""
    total = int(labels.shape[0])
    correct = int(np.sum(predictions == labels))
    accuracy = float(correct / total) if total else 0.0
    return OverallMetrics(correct=correct, total=total, accuracy=accuracy)


def compute_per_class_metrics(
    predictions: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> list[ClassMetrics]:
    """Compute per-class accuracy in class_names order."""
    per_class: list[ClassMetrics] = []
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        class_total = int(np.sum(mask))
        class_correct = int(np.sum(predictions[mask] == class_idx))
        class_acc = float(class_correct / class_total) if class_total else 0.0
        per_class.append(
            ClassMetrics(
                class_name=class_name,
                correct=class_correct,
                total=class_total,
                accuracy=class_acc,
            )
        )
    return per_class
