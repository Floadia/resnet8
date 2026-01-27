#!/usr/bin/env python3
"""Evaluate ONNX ResNet8 model on CIFAR-10 test set."""

import argparse
import os
import pickle

import numpy as np
import onnxruntime as ort


def load_cifar10_test(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CIFAR-10 test batch and class names.

    Args:
        data_dir: Path to cifar-10-batches-py directory

    Returns:
        Tuple of (images, labels, class_names)
        - images: float32 array of shape (10000, 32, 32, 3) normalized to [0, 1]
        - labels: int array of shape (10000,)
        - class_names: list of 10 class name strings
    """
    # Load test batch
    test_path = os.path.join(data_dir, "test_batch")
    with open(test_path, "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    # Extract data using byte-string keys (Python 3 pickle with encoding='bytes')
    raw_images = test_data[b"data"]  # Shape: (10000, 3072)
    labels = np.array(test_data[b"labels"])  # Shape: (10000,)

    # Reshape: (10000, 3072) -> (10000, 3, 32, 32) -> (10000, 32, 32, 3)
    images = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert to float32 WITHOUT normalizing - model was trained on raw pixel values (0-255)
    # See: /mnt/ext1/references/tiny/benchmark/training/image_classification/train.py
    images = images.astype(np.float32)

    # Load class names from batches.meta
    meta_path = os.path.join(data_dir, "batches.meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")

    # Decode byte strings to regular strings
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

    return images, labels, class_names


def evaluate_model(model_path: str, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Run inference on images using ONNX model.

    Args:
        model_path: Path to ONNX model file
        images: float32 array of shape (N, 32, 32, 3)
        labels: int array of shape (N,)

    Returns:
        Predicted class indices array of shape (N,)
    """
    # Create inference session
    session = ort.InferenceSession(model_path)

    # Get input details
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print(f"Model input: {input_name}, shape: {input_shape}")
    print(f"Test images: {images.shape}")

    # Run inference on full batch (model has dynamic batch dimension)
    outputs = session.run(None, {input_name: images})
    logits = outputs[0]  # Shape: (N, 10)

    # Get predicted classes
    predictions = np.argmax(logits, axis=1)

    return predictions


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray,
                     class_names: list[str]) -> tuple[float, dict[str, tuple[int, int, float]]]:
    """Compute overall and per-class accuracy.

    Args:
        predictions: Predicted class indices
        labels: Ground truth labels
        class_names: List of class name strings

    Returns:
        Tuple of (overall_accuracy, per_class_dict)
        - overall_accuracy: float in [0, 1]
        - per_class_dict: {class_name: (correct, total, accuracy)}
    """
    # Overall accuracy
    correct = np.sum(predictions == labels)
    total = len(labels)
    overall_acc = correct / total

    # Per-class accuracy using boolean masking
    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        class_total = np.sum(mask)
        class_correct = np.sum(predictions[mask] == class_idx)
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        per_class[class_name] = (int(class_correct), int(class_total), class_acc)

    return overall_acc, per_class


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX ResNet8 model on CIFAR-10 test set"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.onnx",
        help="Path to ONNX model file (default: models/resnet8.onnx)"
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="Path to cifar-10-batches-py directory"
    )
    args = parser.parse_args()

    # Load CIFAR-10 test data
    print(f"Loading CIFAR-10 test data from: {args.data_dir}")
    images, labels, class_names = load_cifar10_test(args.data_dir)
    print(f"Loaded {len(labels)} test images")
    print(f"Classes: {', '.join(class_names)}")
    print()

    # Run evaluation
    print(f"Loading model: {args.model}")
    predictions = evaluate_model(args.model, images, labels)
    print()

    # Compute accuracy
    overall_acc, per_class = compute_accuracy(predictions, labels, class_names)

    # Print results
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print()

    # Overall accuracy
    correct = int(overall_acc * len(labels))
    print(f"Overall Accuracy: {correct}/{len(labels)} = {overall_acc * 100:.2f}%")
    print()

    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("-" * 40)
    for class_name in class_names:
        class_correct, class_total, class_acc = per_class[class_name]
        print(f"  {class_name:12s}: {class_correct:4d}/{class_total:4d} = {class_acc * 100:5.2f}%")
    print("-" * 40)


if __name__ == "__main__":
    main()
