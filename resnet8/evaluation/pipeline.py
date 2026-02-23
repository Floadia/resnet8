"""Backend-agnostic evaluation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from resnet8.evaluation.adapters import InferenceAdapter
from resnet8.evaluation.cifar10 import load_cifar10_test
from resnet8.evaluation.metrics import (
    compute_overall_metrics,
    compute_per_class_metrics,
)
from resnet8.evaluation.report import build_report


def _predictions_from_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        msg = f"Expected logits with shape [N, C], got {logits.shape}"
        raise ValueError(msg)
    return np.argmax(logits, axis=1).astype(np.int64, copy=False)


def evaluate_arrays(
    *,
    adapter: InferenceAdapter,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    data_dir: str,
) -> dict[str, Any]:
    """Evaluate from in-memory arrays and return canonical report."""
    logits = adapter.predict_logits(images)
    predictions = _predictions_from_logits(logits)

    overall = compute_overall_metrics(predictions, labels)
    per_class = compute_per_class_metrics(predictions, labels, class_names)

    return build_report(
        backend=adapter.backend,
        model_path=adapter.model_path,
        data_dir=data_dir,
        overall=overall,
        per_class=per_class,
    )


def evaluate_dataset(
    *,
    adapter: InferenceAdapter,
    data_dir: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate adapter on CIFAR-10 test dataset."""
    images, labels, class_names = load_cifar10_test(data_dir, max_samples=max_samples)
    return evaluate_arrays(
        adapter=adapter,
        images=images,
        labels=labels,
        class_names=class_names,
        data_dir=data_dir,
    )
