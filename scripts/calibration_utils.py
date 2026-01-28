#!/usr/bin/env python3
"""Calibration data utilities for Post-Training Quantization.

Provides stratified CIFAR-10 calibration samples with preprocessing identical
to evaluation pipeline. Used by ONNX Runtime and PyTorch static quantization.
"""

import argparse
import os
import pickle
from typing import Tuple

import numpy as np


def load_calibration_data(
    data_dir: str, samples_per_class: int = 100
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load stratified CIFAR-10 calibration samples from training batches.

    Args:
        data_dir: Path to cifar-10-batches-py directory
        samples_per_class: Number of samples per class (default: 100)

    Returns:
        Tuple of (images, labels, class_names)
        - images: float32 array of shape (N, 32, 32, 3) with raw pixel values [0, 255]
        - labels: int array of shape (N,)
        - class_names: list of 10 class name strings

    Note:
        Preprocessing matches evaluate.py exactly:
        - Reshape from (N, 3072) to (N, 32, 32, 3) via reshape + transpose
        - Convert to float32 WITHOUT normalization (raw pixel values 0-255)
    """
    # Load class names from batches.meta
    meta_path = os.path.join(data_dir, "batches.meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

    # Collect samples from all training batches (data_batch_1 through data_batch_5)
    all_images = []
    all_labels = []

    for batch_num in range(1, 6):  # data_batch_1 through data_batch_5
        batch_path = os.path.join(data_dir, f"data_batch_{batch_num}")
        with open(batch_path, "rb") as f:
            batch_data = pickle.load(f, encoding="bytes")

        raw_images = batch_data[b"data"]  # Shape: (10000, 3072)
        labels = np.array(batch_data[b"labels"])  # Shape: (10000,)

        all_images.append(raw_images)
        all_labels.append(labels)

    # Concatenate all batches
    all_images = np.concatenate(all_images, axis=0)  # Shape: (50000, 3072)
    all_labels = np.concatenate(all_labels, axis=0)  # Shape: (50000,)

    # Stratified sampling: select exactly samples_per_class from each class
    calibration_images = []
    calibration_labels = []

    for class_idx in range(10):
        # Find all samples of this class
        class_mask = all_labels == class_idx
        class_indices = np.where(class_mask)[0]

        # Sample without replacement
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_idx}: "
                f"requested {samples_per_class}, available {len(class_indices)}"
            )

        selected_indices = np.random.choice(
            class_indices, size=samples_per_class, replace=False
        )

        calibration_images.append(all_images[selected_indices])
        calibration_labels.append(all_labels[selected_indices])

    # Concatenate selected samples
    calibration_images = np.concatenate(calibration_images, axis=0)
    calibration_labels = np.concatenate(calibration_labels, axis=0)

    # Preprocessing: MUST match evaluate.py exactly
    # Reshape: (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
    calibration_images = calibration_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert to float32 WITHOUT normalizing
    # Model was trained on raw pixel values (0-255). See: evaluate.py line 38
    calibration_images = calibration_images.astype(np.float32)

    return calibration_images, calibration_labels, class_names


def verify_distribution(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    """Verify class distribution in calibration dataset.

    Args:
        labels: Array of class labels
        class_names: List of class name strings

    Returns:
        Dictionary mapping class_name to sample count
    """
    distribution = {}
    print("\nClass Distribution:")
    print("-" * 40)

    for class_idx, class_name in enumerate(class_names):
        count = int(np.sum(labels == class_idx))
        distribution[class_name] = count
        print(f"  {class_name:12s}: {count:4d} samples")

    print("-" * 40)

    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Load and verify CIFAR-10 calibration data for PTQ"
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="Path to cifar-10-batches-py directory",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of samples per class (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Load calibration data
    print(f"Loading calibration data from: {args.data_dir}")
    print(f"Samples per class: {args.samples_per_class}")
    print()

    images, labels, class_names = load_calibration_data(
        args.data_dir, args.samples_per_class
    )

    # Print dataset info
    total_samples = len(labels)
    print(f"Loaded {total_samples} calibration samples")
    print()

    # Verify preprocessing matches evaluation
    print("Preprocessing Verification:")
    print("-" * 40)
    print(f"  dtype: {images.dtype}")
    print(f"  shape: {images.shape}")
    print(f"  pixel range: [{images.min():.1f}, {images.max():.1f}]")
    print("  format: NHWC (samples, height, width, channels)")
    print("-" * 40)

    # Verify class distribution
    distribution = verify_distribution(labels, class_names)

    # Sanity checks
    print("\nSanity Checks:")
    print("-" * 40)
    expected_total = args.samples_per_class * 10
    if total_samples == expected_total:
        print(f"✓ Total samples: {total_samples} (expected {expected_total})")
    else:
        print(f"✗ Total samples: {total_samples} (expected {expected_total})")

    if images.dtype == np.float32:
        print("✓ dtype: float32")
    else:
        print(f"✗ dtype: {images.dtype} (expected float32)")

    if images.shape == (total_samples, 32, 32, 3):
        print("✓ shape: NHWC format")
    else:
        print(f"✗ shape: {images.shape} (expected ({total_samples}, 32, 32, 3))")

    if 0 <= images.min() <= 5 and 250 <= images.max() <= 255:
        print("✓ pixel range: [0, 255] (no normalization)")
    else:
        print(
            f"⚠ pixel range: [{images.min():.1f}, {images.max():.1f}]"
            f" (expected [0, 255])"
        )

    all_balanced = all(
        count == args.samples_per_class for count in distribution.values()
    )
    if all_balanced:
        print(f"✓ distribution: balanced ({args.samples_per_class} per class)")
    else:
        print("✗ distribution: imbalanced")

    print("-" * 40)


if __name__ == "__main__":
    main()
