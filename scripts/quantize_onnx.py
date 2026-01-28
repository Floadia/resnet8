#!/usr/bin/env python3
"""Quantize ResNet8 ONNX model to int8 and uint8 using static quantization.

Performs post-training quantization (PTQ) with MinMax calibration using 1000
stratified CIFAR-10 samples. Produces both int8 and uint8 quantized models.
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import onnx

# Add scripts directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_utils import load_calibration_data

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class CIFARCalibrationDataReader(CalibrationDataReader):
    """CalibrationDataReader for CIFAR-10 dataset.

    Wraps calibration_utils.load_calibration_data() to provide iterator interface
    required by ONNX Runtime quantize_static API.

    Note: Iterator is consumed after use. Create fresh instance for each quantization.
    """

    def __init__(self, model_path: str, data_dir: str, samples_per_class: int = 100):
        """Initialize calibration data reader.

        Args:
            model_path: Path to ONNX model (to extract input name)
            data_dir: Path to cifar-10-batches-py directory
            samples_per_class: Number of calibration samples per class
        """
        # Load calibration data using stratified sampling
        self.images, self.labels, self.class_names = load_calibration_data(
            data_dir, samples_per_class
        )

        # Get model input name from ONNX metadata
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name

        # Iterator state
        self.current_index = 0
        self.total_samples = len(self.images)

    def get_next(self) -> Dict[str, np.ndarray] | None:
        """Return next calibration sample.

        Returns:
            Dictionary with single sample: {input_name: array of shape (1, 32, 32, 3)}
            or None when exhausted
        """
        if self.current_index >= self.total_samples:
            return None

        # Get single sample and add batch dimension
        sample = self.images[self.current_index]  # Shape: (32, 32, 3)
        batch_sample = np.expand_dims(sample, axis=0)  # Shape: (1, 32, 32, 3)

        self.current_index += 1

        return {self.input_name: batch_sample}

    def rewind(self) -> None:
        """Reset iterator to beginning (optional method for debugging)."""
        self.current_index = 0


def ensure_onnx_model(model_path: str) -> None:
    """Ensure ONNX model exists, run conversion if missing.

    Args:
        model_path: Path to ONNX model file
    """
    if os.path.exists(model_path):
        print(f"✓ ONNX model exists: {model_path}")
        return

    print(f"⚠ ONNX model not found: {model_path}")
    print("  Running conversion script...")

    # Run convert.py from scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(scripts_dir, "convert.py")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")

    import subprocess

    result = subprocess.run(
        ["python", convert_script], cwd=os.path.dirname(scripts_dir), check=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Model conversion failed with exit code {result.returncode}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Conversion completed but model not found: {model_path}"
        )

    print(f"✓ ONNX model created: {model_path}")


def quantize_model(
    model_path: str,
    output_int8: str,
    output_uint8: str,
    data_dir: str,
    samples_per_class: int,
    seed: int,
) -> None:
    """Quantize ONNX model to int8 and uint8 using static quantization.

    Args:
        model_path: Source ONNX model path
        output_int8: Int8 quantized model output path
        output_uint8: Uint8 quantized model output path
        data_dir: CIFAR-10 data directory
        samples_per_class: Number of calibration samples per class
        seed: Random seed for reproducible calibration sampling
    """
    # Ensure base ONNX model exists
    ensure_onnx_model(model_path)

    # Set random seed for reproducible calibration
    np.random.seed(seed)
    print(f"\nQuantization Configuration:")
    print("-" * 50)
    print(f"  Random seed: {seed}")
    print(f"  Source model: {model_path}")
    print(f"  Calibration samples: {samples_per_class * 10} ({samples_per_class} per class)")
    print(f"  Calibration method: MinMax")
    print(f"  Quantization format: QDQ (recommended for CPU)")
    print(f"  Per-channel quantization: False")
    print("-" * 50)
    print()

    # Create output directory if needed
    output_dir = os.path.dirname(output_int8)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Quantize to Int8 (QInt8)
    print("Quantizing to Int8 (QuantType.QInt8)...")
    print(f"  Output: {output_int8}")

    # Create fresh calibration reader (iterator will be consumed)
    calibration_reader_int8 = CIFARCalibrationDataReader(
        model_path, data_dir, samples_per_class
    )

    quantize_static(
        model_input=model_path,
        model_output=output_int8,
        calibration_data_reader=calibration_reader_int8,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )
    print(f"✓ Int8 quantization complete")
    print()

    # Quantize to Uint8 (QUInt8)
    print("Quantizing to Uint8 (QuantType.QUInt8)...")
    print(f"  Output: {output_uint8}")

    # Create NEW calibration reader (previous one was consumed)
    calibration_reader_uint8 = CIFARCalibrationDataReader(
        model_path, data_dir, samples_per_class
    )

    quantize_static(
        model_input=model_path,
        model_output=output_uint8,
        calibration_data_reader=calibration_reader_uint8,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )
    print(f"✓ Uint8 quantization complete")
    print()

    # Summary
    print("=" * 50)
    print("QUANTIZATION COMPLETE")
    print("=" * 50)
    print(f"Source model:  {model_path}")
    print(f"Int8 model:    {output_int8}")
    print(f"Uint8 model:   {output_uint8}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ResNet8 ONNX model to int8 and uint8 using static quantization"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8.onnx",
        help="Source ONNX model path (default: models/resnet8.onnx)",
    )
    parser.add_argument(
        "--output-int8",
        default="models/resnet8_int8.onnx",
        help="Int8 output path (default: models/resnet8_int8.onnx)",
    )
    parser.add_argument(
        "--output-uint8",
        default="models/resnet8_uint8.onnx",
        help="Uint8 output path (default: models/resnet8_uint8.onnx)",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="CIFAR-10 data directory",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Calibration samples per class (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    quantize_model(
        model_path=args.model,
        output_int8=args.output_int8,
        output_uint8=args.output_uint8,
        data_dir=args.data_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
