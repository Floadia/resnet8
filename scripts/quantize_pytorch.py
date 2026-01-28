#!/usr/bin/env python3
"""Quantize ResNet8 PyTorch model using static quantization (eager mode).

Uses torch.ao.quantization APIs with fbgemm backend for CPU inference.
Calibration data loaded from calibration_utils.py (1000 stratified samples).
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader, TensorDataset

# Add scripts directory to path for calibration_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_utils import load_calibration_data


def load_pytorch_model(model_path: str) -> torch.nn.Module:
    """Load PyTorch model from checkpoint.

    Args:
        model_path: Path to .pt model file

    Returns:
        PyTorch model in eval mode
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()  # CRITICAL: must be in eval mode for quantization
    return model


def inspect_model_structure(model: torch.nn.Module) -> None:
    """Print model structure to identify module types and names.

    Args:
        model: PyTorch model to inspect
    """
    print("\nModel Structure:")
    print("=" * 70)
    for name, module in model.named_modules():
        print(f"{name:40s} {type(module).__name__}")
    print("=" * 70)


def create_calibration_loader(
    data_dir: str, samples_per_class: int = 100, batch_size: int = 32
) -> DataLoader:
    """Create calibration DataLoader from existing utilities.

    Args:
        data_dir: Path to cifar-10-batches-py directory
        samples_per_class: Number of samples per class (default: 100)
        batch_size: Batch size for calibration (default: 32)

    Returns:
        DataLoader with calibration samples
    """
    # Set random seed for reproducibility (matches calibration_utils default)
    np.random.seed(42)

    # Load stratified calibration data
    images, labels, _ = load_calibration_data(data_dir, samples_per_class)

    print(f"Loaded {len(images)} calibration samples")
    print(f"  Shape: {images.shape}")
    print(f"  dtype: {images.dtype}")
    print(f"  Range: [{images.min():.1f}, {images.max():.1f}]")
    print(f"  Format: NHWC (samples, height, width, channels)")

    # Convert to tensors - keep NHWC format (model expects this from evaluate_pytorch.py)
    dataset = TensorDataset(
        torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Created DataLoader with batch_size={batch_size}, {len(loader)} batches")

    return loader


def quantize_model_eager(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
) -> torch.nn.Module:
    """Quantize model using eager mode static quantization.

    Uses fbgemm backend with default qconfig (quint8 activations, qint8 weights).
    Does NOT apply layer fusion initially (onnx2torch model structure unknown).

    Args:
        model: FP32 PyTorch model in eval mode
        calibration_loader: DataLoader with calibration samples

    Returns:
        Quantized PyTorch model
    """
    # Step 1: Skip fusion initially (onnx2torch model structure unknown)
    # Layer fusion can be added later if needed based on model inspection
    print("\nSkipping fusion (starting with baseline, no fusion patterns)")

    # Step 2: Configure quantization
    print("\nConfiguring quantization (fbgemm backend)...")
    model.qconfig = get_default_qconfig("fbgemm")
    print(f"  Activation observer: {model.qconfig.activation}")
    print(f"  Weight observer: {model.qconfig.weight}")

    # Step 3: Prepare model (insert observers)
    print("\nPreparing model (inserting observers)...")
    prepared_model = prepare(model, inplace=False)
    print("Model prepared")

    # Step 4: Calibration
    print(f"\nCalibrating with {len(calibration_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            prepared_model(data)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(calibration_loader)} batches")
    print("Calibration complete")

    # Step 5: Convert to quantized model
    print("\nConverting to quantized model...")
    quantized_model = convert(prepared_model, inplace=False)
    print("Conversion complete")

    return quantized_model


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ResNet8 PyTorch model using static quantization (eager mode)"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.pt",
        help="Path to FP32 PyTorch model (default: models/resnet8.pt)",
    )
    parser.add_argument(
        "--output",
        default="models/resnet8_int8.pt",
        help="Path for quantized model output (default: models/resnet8_int8.pt)",
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
        help="Calibration samples per class (default: 100, total: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Calibration batch size (default: 32)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect model structure without quantizing",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PYTORCH STATIC QUANTIZATION (EAGER MODE)")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_pytorch_model(args.model)
    print("Model loaded and set to eval mode")

    # Inspect structure
    inspect_model_structure(model)

    if args.inspect_only:
        print("\nInspection complete (--inspect-only flag set)")
        return

    # Create calibration loader
    print(f"\nLoading calibration data from: {args.data_dir}")
    calibration_loader = create_calibration_loader(
        args.data_dir, args.samples_per_class, args.batch_size
    )

    # Quantize model
    quantized_model = quantize_model_eager(model, calibration_loader)

    # Save quantized model (same format as evaluate_pytorch.py expects)
    print(f"\nSaving quantized model to: {args.output}")
    torch.save({"model": quantized_model}, args.output)
    print("Quantized model saved")

    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"Output: {args.output}")
    print("\nNext step: Evaluate with scripts/evaluate_pytorch.py --model " + args.output)


if __name__ == "__main__":
    main()
