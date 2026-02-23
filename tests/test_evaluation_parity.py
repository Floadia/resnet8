from __future__ import annotations

import numpy as np

from resnet8.evaluation.pipeline import evaluate_arrays


class _DummyAdapter:
    def __init__(self, backend: str, model_path: str, logits: np.ndarray):
        self.backend = backend
        self.model_path = model_path
        self._logits = logits

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        assert images.dtype == np.float32
        return self._logits


def test_backend_adapters_share_metric_pipeline():
    images = np.zeros((4, 32, 32, 3), dtype=np.float32)
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    class_names = ["zero", "one"]
    logits = np.array(
        [
            [9.0, 1.0],
            [1.0, 9.0],
            [2.0, 8.0],
            [8.0, 2.0],
        ],
        dtype=np.float32,
    )

    onnx_report = evaluate_arrays(
        adapter=_DummyAdapter("onnx", "models/resnet8.onnx", logits),
        images=images,
        labels=labels,
        class_names=class_names,
        data_dir="/tmp/cifar",
    )
    pytorch_report = evaluate_arrays(
        adapter=_DummyAdapter("pytorch", "models/resnet8.pt", logits),
        images=images,
        labels=labels,
        class_names=class_names,
        data_dir="/tmp/cifar",
    )

    assert onnx_report["overall"] == pytorch_report["overall"]
    assert onnx_report["per_class"] == pytorch_report["per_class"]
    assert onnx_report["overall"]["accuracy"] == 1.0
