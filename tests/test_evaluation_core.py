from __future__ import annotations

import pickle

import numpy as np

from resnet8.evaluation.cifar10 import load_cifar10_test
from resnet8.evaluation.metrics import (
    compute_overall_metrics,
    compute_per_class_metrics,
)
from resnet8.evaluation.report import build_report, format_report_text


def _write_cifar_fixture(tmp_path) -> str:
    data_dir = tmp_path / "cifar"
    data_dir.mkdir(parents=True)

    raw = np.arange(4 * 3072, dtype=np.uint8).reshape(4, 3072)
    labels = [0, 1, 1, 0]

    with (data_dir / "test_batch").open("wb") as f:
        pickle.dump({b"data": raw, b"labels": labels}, f)

    with (data_dir / "batches.meta").open("wb") as f:
        pickle.dump({b"label_names": [b"zero", b"one"]}, f)

    return str(data_dir)


def test_load_cifar10_test_shared_preprocessing(tmp_path):
    data_dir = _write_cifar_fixture(tmp_path)

    images, labels, class_names = load_cifar10_test(data_dir, max_samples=3)

    assert images.shape == (3, 32, 32, 3)
    assert images.dtype == np.float32
    assert labels.tolist() == [0, 1, 1]
    assert class_names == ["zero", "one"]
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 255.0


def test_metrics_and_report_schema_are_deterministic():
    predictions = np.array([0, 1, 0, 0], dtype=np.int64)
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    class_names = ["zero", "one"]

    overall = compute_overall_metrics(predictions, labels)
    per_class = compute_per_class_metrics(predictions, labels, class_names)

    report = build_report(
        backend="pytorch",
        model_path="models/resnet8.pt",
        data_dir="/tmp/cifar",
        overall=overall,
        per_class=per_class,
    )

    assert list(report.keys()) == [
        "schema_version",
        "backend",
        "model_path",
        "data_dir",
        "overall",
        "per_class",
    ]
    assert report["schema_version"] == "1.0"
    assert report["overall"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert report["per_class"][0]["class_name"] == "zero"
    assert report["per_class"][1]["class_name"] == "one"

    text = format_report_text(report, title="EVALUATION RESULTS")
    assert "Overall Accuracy: 3/4 = 75.00%" in text
    assert "Per-Class Accuracy:" in text
