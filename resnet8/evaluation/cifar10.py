"""CIFAR-10 dataset loading for evaluation workflows."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

DEFAULT_CIFAR10_DATA_DIR = (
    "/mnt/ext1/references/tiny/benchmark/training/"
    "image_classification/cifar-10-batches-py"
)


def _decode_label_names(raw_names: list[bytes]) -> list[str]:
    return [name.decode("utf-8") for name in raw_names]


def _load_pickle_bytes(path: Path) -> dict[bytes, object]:
    visible_warning = getattr(np, "VisibleDeprecationWarning", None)
    if visible_warning is None and hasattr(np, "exceptions"):
        visible_warning = getattr(np.exceptions, "VisibleDeprecationWarning", None)

    with path.open("rb") as f:
        with warnings.catch_warnings():
            if visible_warning is not None:
                warnings.filterwarnings("ignore", category=visible_warning)
            return pickle.load(f, encoding="bytes")


def load_cifar10_test(
    data_dir: str | Path, max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CIFAR-10 test images, labels, and class names.

    Images are returned in NHWC layout with raw pixel values [0, 255] as float32.
    """
    data_path = Path(data_dir)

    test_batch_path = data_path / "test_batch"
    test_data = _load_pickle_bytes(test_batch_path)

    raw_images = test_data[b"data"]
    labels = np.asarray(test_data[b"labels"], dtype=np.int64)
    images = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = images.astype(np.float32, copy=False)

    meta_path = data_path / "batches.meta"
    meta = _load_pickle_bytes(meta_path)

    class_names = _decode_label_names(meta[b"label_names"])

    if max_samples is not None:
        if max_samples <= 0:
            msg = "max_samples must be greater than 0"
            raise ValueError(msg)
        images = images[:max_samples]
        labels = labels[:max_samples]

    return images, labels, class_names
