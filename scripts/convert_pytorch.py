#!/usr/bin/env python3
"""Convert ONNX ResNet8 model to PyTorch using onnx2torch."""

import argparse
import os

import torch
from onnx2torch import convert


def convert_onnx_to_pytorch(input_path: str, output_path: str) -> None:
    """Convert ONNX model to PyTorch and save.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save PyTorch model
    """
    print(f"Loading ONNX model from: {input_path}")

    # Convert ONNX to PyTorch
    print("Converting to PyTorch...")
    model = convert(input_path)

    # Set to eval mode (important for BatchNorm)
    model.eval()
    print("Model converted successfully")

    # Verify model structure with test input
    print("\nVerifying model structure...")
    # Input shape: (batch, height, width, channels) = (1, 32, 32, 3)
    test_input = torch.randn(1, 32, 32, 3)

    with torch.no_grad():
        output = model(test_input)

    print(f"  Input shape: {tuple(test_input.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Save model
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # Save both full model and state_dict for flexibility
    torch.save(
        {
            "model": model,
            "state_dict": model.state_dict(),
            "input_shape": (None, 32, 32, 3),  # batch, H, W, C
            "output_shape": (None, 10),  # batch, classes
        },
        output_path,
    )

    print(f"\nSaved PyTorch model to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX ResNet8 model to PyTorch"
    )
    parser.add_argument(
        "--input",
        default="models/resnet8.onnx",
        help="Path to input ONNX model (default: models/resnet8.onnx)",
    )
    parser.add_argument(
        "--output",
        default="models/resnet8.pt",
        help="Path to save PyTorch model (default: models/resnet8.pt)",
    )
    args = parser.parse_args()

    convert_onnx_to_pytorch(args.input, args.output)


if __name__ == "__main__":
    main()
