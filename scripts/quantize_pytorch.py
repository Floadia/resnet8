#!/usr/bin/env python3
"""Quantize ResNet8 PyTorch model using static quantization.

Supports two quantization modes:
1. FX Graph Mode with JIT tracing - Traces model to enable serialization
2. Eager Mode - Requires standard PyTorch module types (limited onnx2torch support)

Uses torch.ao.quantization APIs with fbgemm backend for CPU inference.
Calibration data loaded from calibration_utils.py (1000 stratified samples).

IMPORTANT: onnx2torch models use custom ONNX operations that have limited
support for PyTorch quantization. This script implements a best-effort approach
using FX graph mode with JIT tracing to enable model serialization.

NOTE: fbgemm backend supports only int8 quantization (quint8 activations + qint8 weights).
uint8-only quantization (like ONNX Runtime's U8U8 mode) is NOT supported - PyTorch's
quantized convolution requires qint8 (signed) weights. The "int8" output from this
script uses unsigned 8-bit activations (quint8) and signed 8-bit weights (qint8).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig
from torch.utils.data import DataLoader, TensorDataset

# Suppress deprecation warnings for torch.ao.quantization (will migrate to torchao later)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao.quantization")

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


def quantize_model_fx(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    output_path: str,
) -> torch.jit.ScriptModule:
    """Quantize model using FX graph mode static quantization with JIT tracing.

    FX mode works with onnx2torch models by tracing the computation graph.
    JIT tracing is used for serialization since FX GraphModule has issues.

    Args:
        model: FP32 PyTorch model in eval mode
        calibration_loader: DataLoader with calibration samples
        output_path: Path to save quantized model

    Returns:
        JIT traced quantized model
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    # Get example input for tracing
    example_input = next(iter(calibration_loader))[0][:1]  # Single sample
    print(f"\nExample input shape: {tuple(example_input.shape)}")

    # Configure quantization using proper QConfigMapping API
    print("\nConfiguring quantization (fbgemm backend, FX mode)...")
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    print(f"  Backend: fbgemm")
    print(f"  Using default qconfig mapping (HistogramObserver for activations, PerChannelMinMaxObserver for weights)")

    # Prepare model (trace graph and insert observers)
    print("\nPreparing model (FX graph tracing and inserting observers)...")
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=(example_input,))
    print("Model prepared")

    # Calibration
    print(f"\nCalibrating with {len(calibration_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            prepared_model(data)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(calibration_loader)} batches")
    print("Calibration complete")

    # Convert to quantized model
    print("\nConverting to quantized model...")
    quantized_model = convert_fx(prepared_model)
    print("Conversion complete")

    # JIT trace for serialization
    print("\nTracing model with JIT for serialization...")
    quantized_model.eval()
    traced_model = torch.jit.trace(quantized_model, example_input)
    print("JIT tracing complete")

    # Save as TorchScript
    print(f"\nSaving TorchScript model to: {output_path}")
    traced_model.save(output_path)
    print("Model saved")

    return traced_model


def quantize_model_eager(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
) -> torch.nn.Module:
    """Quantize model using eager mode static quantization.

    Note: Eager mode requires standard PyTorch module types (Conv2d, Linear, etc.)
    and may not work with onnx2torch models that use custom ONNX operations.

    Args:
        model: FP32 PyTorch model in eval mode
        calibration_loader: DataLoader with calibration samples

    Returns:
        Quantized PyTorch model
    """
    from torch.ao.quantization import prepare, convert

    print("\nNote: Eager mode may not work with onnx2torch models")
    print("Use --mode fx for FX graph mode quantization (recommended)")

    # Configure quantization
    print("\nConfiguring quantization (fbgemm backend, eager mode)...")
    model.qconfig = get_default_qconfig("fbgemm")
    print(f"  Activation observer: {model.qconfig.activation}")
    print(f"  Weight observer: {model.qconfig.weight}")

    # Prepare model (insert observers)
    print("\nPreparing model (inserting observers)...")
    prepared_model = prepare(model, inplace=False)
    print("Model prepared")

    # Calibration
    print(f"\nCalibrating with {len(calibration_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            prepared_model(data)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(calibration_loader)} batches")
    print("Calibration complete")

    # Convert to quantized model
    print("\nConverting to quantized model...")
    quantized_model = convert(prepared_model, inplace=False)
    print("Conversion complete")

    return quantized_model


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ResNet8 PyTorch model using static quantization"
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
        "--mode",
        choices=["fx", "eager"],
        default="fx",
        help="Quantization mode: fx (FX graph, default) or eager",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect model structure without quantizing",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"PYTORCH STATIC QUANTIZATION ({args.mode.upper()} MODE)")
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

    # Quantize model using selected mode
    if args.mode == "fx":
        # FX mode saves model directly (uses JIT tracing for serialization)
        quantize_model_fx(model, calibration_loader, args.output)
    else:
        quantized_model = quantize_model_eager(model, calibration_loader)
        # Save quantized model (same format as evaluate_pytorch.py expects)
        print(f"\nSaving quantized model to: {args.output}")
        torch.save({"model": quantized_model}, args.output)
        print("Quantized model saved")

    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"Output: {args.output}")
    if args.mode == "fx":
        print("\nNote: FX mode saves TorchScript model (load with torch.jit.load)")
    print("\nNext step: Evaluate with scripts/evaluate_pytorch.py --model " + args.output)


if __name__ == "__main__":
    main()
