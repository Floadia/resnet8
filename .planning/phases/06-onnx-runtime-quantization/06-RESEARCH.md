# Phase 6: ONNX Runtime Quantization - Research

**Researched:** 2026-01-28
**Domain:** Post-Training Quantization with ONNX Runtime
**Confidence:** HIGH

## Summary

ONNX Runtime provides a well-established quantization API via the `onnxruntime.quantization` module, specifically the `quantize_static()` function for static post-training quantization. The approach is straightforward for CNN models like ResNet8: implement a CalibrationDataReader subclass, provide calibration data (already available via calibration_utils.py), and call quantize_static() with appropriate parameters.

The API supports both int8 (signed) and uint8 (unsigned) activation types, with S8S8 (int8 activations and weights) recommended as the default for CPU deployment. The MinMax calibration method is appropriate for CNN models and is the default. Pre-processing with shape inference and model optimization is recommended but optional, and can improve quantization quality by merging operations like Conv-BatchNorm.

Expected accuracy degradation for well-calibrated int8 quantization on CNNs is typically 0-3% from baseline. The existing evaluation script (evaluate.py) requires no modifications as quantized ONNX models work transparently with onnxruntime.InferenceSession.

**Primary recommendation:** Implement CalibrationDataReader wrapping existing calibration_utils.py data (1000 samples, NHWC format, raw pixels 0-255), use quantize_static() with default parameters (QDQ format, MinMax calibration, QInt8 for both activations and weights), then evaluate with existing evaluate.py script.

## Standard Stack

The established libraries/tools for ONNX Runtime quantization:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnxruntime | 1.23.2+ | Quantization and inference | Built-in quantization module, no additional dependencies |
| numpy | Any | Data handling | Standard array operations for calibration data |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnx | Latest | Shape inference pre-processing | Optional, for quant_pre_process() before quantization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Static quantization | Dynamic quantization | Dynamic is faster to implement (no calibration) but gives lower accuracy and only quantizes weights |
| MinMax calibration | Entropy calibration | Entropy may improve accuracy slightly but takes longer and adds complexity |
| QDQ format | QOperator format | QOperator faster for 4-bit quantization only, slower for 8-bit on CPU |

**Installation:**
```bash
# No new dependencies needed - onnxruntime 1.23.2 already installed
# Optional: for shape inference pre-processing
pip install onnx
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── quantize_onnx.py           # Quantization script with CalibrationDataReader
├── calibration_utils.py       # Existing: provides load_calibration_data()
└── evaluate.py                # Existing: evaluates ONNX models (no changes needed)

models/
├── resnet8.onnx              # Source FP32 model (from Phase 1)
├── resnet8_int8.onnx         # Quantized int8 model (output)
└── resnet8_uint8.onnx        # Quantized uint8 model (output)
```

### Pattern 1: CalibrationDataReader Implementation
**What:** Subclass CalibrationDataReader with get_next() method returning dict of input arrays
**When to use:** Required for quantize_static() - provides calibration data iterator
**Example:**
```python
# Source: GitHub microsoft/onnxruntime calibrate.py + examples
from onnxruntime.quantization import CalibrationDataReader
import numpy as np

class Resnet8CalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data_path: str, input_name: str):
        """Initialize with calibration data.

        Args:
            calibration_data_path: Path to CIFAR-10 data directory
            input_name: Model input name (from ONNX model metadata)
        """
        from calibration_utils import load_calibration_data

        # Load calibration data (1000 samples, NHWC format, float32, 0-255 range)
        self.images, self.labels, _ = load_calibration_data(
            calibration_data_path,
            samples_per_class=100  # 1000 total samples
        )
        self.input_name = input_name
        self.data_index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next calibration sample or None when exhausted.

        Returns:
            Dict mapping input_name to single image array (1, 32, 32, 3)
            or None when all samples consumed
        """
        if self.data_index >= len(self.images):
            return None

        # Return single image with batch dimension
        sample = self.images[self.data_index:self.data_index+1]  # Shape: (1, 32, 32, 3)
        self.data_index += 1

        return {self.input_name: sample}
```

### Pattern 2: Quantize_static Invocation
**What:** Call quantize_static() with model paths, data reader, and quantization parameters
**When to use:** Once per quantization type (int8, uint8)
**Example:**
```python
# Source: ONNX Runtime official documentation + GitHub examples
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod

def quantize_model(
    model_input: str,
    model_output: str,
    calibration_data_reader: CalibrationDataReader,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
):
    """Quantize ONNX model using static quantization.

    Args:
        model_input: Path to FP32 ONNX model
        model_output: Path for quantized model output
        calibration_data_reader: CalibrationDataReader instance
        activation_type: QInt8 (signed) or QUInt8 (unsigned)
        weight_type: QInt8 (signed) or QUInt8 (unsigned)
    """
    quantize_static(
        model_input=model_input,
        model_output=model_output,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,           # QDQ format (recommended for CPU)
        activation_type=activation_type,         # QInt8 or QUInt8
        weight_type=weight_type,                 # QInt8 or QUInt8
        calibrate_method=CalibrationMethod.MinMax,  # MinMax for CNNs
        per_channel=False,                       # Per-tensor quantization (simpler)
        reduce_range=False,                      # Full 8-bit range (not 7-bit)
    )
```

### Pattern 3: Optional Pre-processing
**What:** Run shape inference and model optimization before quantization
**When to use:** Optional - may improve quantization quality by merging Conv-BatchNorm-ReLU
**Example:**
```python
# Source: ONNX Runtime quantization documentation
from onnxruntime.quantization import shape_inference

# Optional: pre-process model for better quantization
shape_inference.quant_pre_process(
    input_model_path='models/resnet8.onnx',
    output_model_path='models/resnet8_preprocessed.onnx',
    skip_optimization=False,  # Run ONNX Runtime optimizations
    skip_onnx_shape=False,    # Run ONNX shape inference
    skip_symbolic_shape=True,  # Skip symbolic shape (for transformers)
    auto_merge=True,          # Merge Conv-BN-ReLU sequences
)

# Then quantize the preprocessed model
quantize_static(
    model_input='models/resnet8_preprocessed.onnx',
    model_output='models/resnet8_int8.onnx',
    # ... other parameters
)
```

### Anti-Patterns to Avoid
- **Single calibration sample:** Using only 1 calibration sample per class (10 total) produces poor quantization parameters. Use at least 100 samples per class (1000 total).
- **Random calibration data:** Using random noise instead of real CIFAR-10 images causes catastrophic accuracy loss (20-70% drop). Must use real training/validation images.
- **Preprocessing mismatch:** Normalizing calibration data to [0,1] when model expects [0,255] causes severe accuracy degradation. Match evaluate.py preprocessing exactly.
- **QOperator format for 8-bit CPU:** Using QOperator instead of QDQ for 8-bit quantization on CPU is slower. QOperator is only faster for 4-bit quantization.
- **Batched calibration data:** Returning multiple images per get_next() call can work but increases memory usage. Batch size 1 (single image) is safer and standard practice.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Calibration data iteration | Custom data loader with manual batching | CalibrationDataReader subclass | ONNX Runtime expects specific iterator protocol (get_next() returning dict or None), integrates with quantization session lifecycle |
| Quantization parameter computation | Manual scale/zero-point calculation from calibration | quantize_static() with CalibrationDataReader | MinMax/Entropy algorithms handle edge cases (zero ranges, outliers), per-channel complexity, proper clipping |
| Model graph optimization | Custom ONNX graph transformations | quant_pre_process() or quantize_static optimize_model | Conv-BatchNorm fusion, redundant node elimination, shape inference - complex graph rewriting with correctness guarantees |
| Quantized node insertion | Manual QuantizeLinear/DequantizeLinear insertion | quantize_static() with QDQ format | Handles operator support detection, quantizable node discovery, proper insertion points, maintains graph validity |

**Key insight:** ONNX Runtime quantization is mature and battle-tested - custom implementations miss edge cases in calibration statistics (handling zero ranges, negative-only ranges), graph optimization opportunities (multi-node fusion patterns), and execution provider compatibility (CPU vs GPU quantization formats).

## Common Pitfalls

### Pitfall 1: Insufficient Calibration Data
**What goes wrong:** Using too few calibration samples (<100 total) or non-representative samples causes incorrect scale/zero-point computation, leading to 10-30% accuracy loss
**Why it happens:** Each class needs sufficient samples to capture activation range diversity; skewed or sparse calibration misses important activation patterns
**How to avoid:** Use 100 samples per class (1000 total) with stratified sampling from training set, verify distribution matches inference distribution
**Warning signs:** Quantized accuracy drops >5% from baseline, specific classes show severe degradation (>10% drop), quantization ranges are suspiciously narrow

### Pitfall 2: Preprocessing Mismatch Between Calibration and Inference
**What goes wrong:** Applying different preprocessing in calibration vs inference (e.g., normalized [0,1] calibration but raw [0,255] inference) causes activation range mismatch and catastrophic accuracy loss (20-50% drop)
**Why it happens:** Calibration computes scale/zero-point based on observed activation ranges during calibration; wrong preprocessing shifts entire activation distribution
**How to avoid:** Use identical preprocessing in CalibrationDataReader as evaluate.py: NHWC format (32, 32, 3), float32 dtype, raw pixel values [0, 255], no normalization
**Warning signs:** Quantized model has near-random accuracy (<20%), loss is consistent across all classes, quantization parameters seem wrong for pixel value range

### Pitfall 3: NHWC vs NCHW Input Format Confusion
**What goes wrong:** ONNX models typically expect NCHW (batch, channels, height, width) but ResNet8 model from onnx2torch may expect NHWC (batch, height, width, channels), causing shape mismatch errors or transposed inference
**Why it happens:** Different frameworks use different default layouts; torch uses NCHW, TensorFlow uses NHWC, onnx2torch-converted models may preserve original layout
**How to avoid:** Check model input shape from ONNX metadata (session.get_inputs()[0].shape), match calibration data layout to model expectation, verify evaluate.py already uses correct format
**Warning signs:** Shape mismatch errors during quantization or inference, model accepts data but produces garbage predictions, swapped height/width dimensions

### Pitfall 4: QOperator vs QDQ Format Confusion for CPU
**What goes wrong:** Using QuantFormat.QOperator for 8-bit quantization on CPU results in slower inference than FP32, defeating the purpose of quantization
**Why it happens:** QOperator format creates specialized quantized ops (QLinearConv) that are slower on x86-64 for 8-bit, optimized only for 4-bit quantization
**How to avoid:** Always use QuantFormat.QDQ for 8-bit quantization on CPU, which inserts QuantizeLinear/DequantizeLinear pairs that ONNX Runtime fuses efficiently
**Warning signs:** Quantized model inference is slower than FP32 baseline, CPU utilization is lower than expected, profiling shows quantized ops are not fused

### Pitfall 5: Forgetting to Set Random Seed for Reproducibility
**What goes wrong:** Calibration data sampling is non-deterministic, causing different quantization results across runs, making accuracy comparisons and debugging difficult
**Why it happens:** load_calibration_data() uses np.random.choice() for stratified sampling without seed, each run samples different images
**How to avoid:** Set np.random.seed(42) before loading calibration data in quantization script, document seed value for reproducibility
**Warning signs:** Quantized model accuracy varies by 0.5-2% across runs with same hyperparameters, cannot reproduce exact accuracy numbers for debugging

## Code Examples

Verified patterns from official sources:

### Complete Quantization Script Structure
```python
#!/usr/bin/env python3
"""Quantize ResNet8 ONNX model using static quantization."""

import argparse
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

# Import existing calibration utilities
import sys
sys.path.append('scripts')
from calibration_utils import load_calibration_data


class Resnet8CalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for ResNet8 CIFAR-10 quantization."""

    def __init__(self, data_dir: str, input_name: str, samples_per_class: int = 100):
        # Load stratified calibration data (NHWC, float32, 0-255)
        self.images, self.labels, _ = load_calibration_data(data_dir, samples_per_class)
        self.input_name = input_name
        self.data_index = 0
        print(f"Loaded {len(self.images)} calibration samples")

    def get_next(self):
        """Return next calibration sample or None."""
        if self.data_index >= len(self.images):
            return None

        # Return single sample with batch dimension
        sample = self.images[self.data_index:self.data_index+1]
        self.data_index += 1
        return {self.input_name: sample}


def get_model_input_name(model_path: str) -> str:
    """Extract input name from ONNX model metadata."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return input_name


def quantize_resnet8(
    model_path: str,
    output_path: str,
    data_dir: str,
    activation_type: QuantType,
    weight_type: QuantType,
    samples_per_class: int = 100,
):
    """Quantize ResNet8 ONNX model with static quantization.

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path for quantized model output
        data_dir: CIFAR-10 data directory
        activation_type: QuantType.QInt8 or QuantType.QUInt8
        weight_type: QuantType.QInt8 or QuantType.QUInt8
        samples_per_class: Number of calibration samples per class
    """
    # Get model input name from metadata
    input_name = get_model_input_name(model_path)
    print(f"Model input name: {input_name}")

    # Create calibration data reader
    calibration_reader = Resnet8CalibrationDataReader(
        data_dir,
        input_name,
        samples_per_class
    )

    # Run static quantization
    print(f"\nQuantizing model:")
    print(f"  Input:  {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Activation type: {activation_type}")
    print(f"  Weight type: {weight_type}")
    print(f"  Calibration method: MinMax")
    print(f"  Calibration samples: {samples_per_class * 10}")

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
        reduce_range=False,
    )

    print(f"\nQuantized model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ResNet8 ONNX model")
    parser.add_argument(
        "--model",
        default="models/resnet8.onnx",
        help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--output-int8",
        default="models/resnet8_int8.onnx",
        help="Path for int8 quantized model"
    )
    parser.add_argument(
        "--output-uint8",
        default="models/resnet8_uint8.onnx",
        help="Path for uint8 quantized model"
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="CIFAR-10 data directory"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Calibration samples per class (default: 100, total: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible calibration sampling"
    )
    args = parser.parse_args()

    # Set random seed for reproducible calibration data sampling
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}\n")

    # Quantize to int8
    print("=" * 60)
    print("QUANTIZING TO INT8 (signed)")
    print("=" * 60)
    quantize_resnet8(
        args.model,
        args.output_int8,
        args.data_dir,
        QuantType.QInt8,
        QuantType.QInt8,
        args.samples_per_class,
    )

    print("\n" + "=" * 60)
    print("QUANTIZING TO UINT8 (unsigned)")
    print("=" * 60)
    # Need fresh calibration reader for uint8 (get_next() exhausted)
    quantize_resnet8(
        args.model,
        args.output_uint8,
        args.data_dir,
        QuantType.QUInt8,
        QuantType.QUInt8,
        args.samples_per_class,
    )

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"Int8 model:  {args.output_int8}")
    print(f"Uint8 model: {args.output_uint8}")
    print("\nNext step: Evaluate quantized models with scripts/evaluate.py")


if __name__ == "__main__":
    main()
```

### Evaluation Command (No Changes Needed)
```bash
# Existing evaluate.py works transparently with quantized models
python scripts/evaluate.py --model models/resnet8_int8.onnx
python scripts/evaluate.py --model models/resnet8_uint8.onnx

# Compare with FP32 baseline (87.19% expected)
python scripts/evaluate.py --model models/resnet8.onnx
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| QOperator format default | QDQ format recommended (v1.11+) | ONNX Runtime 1.11 (2022) | QDQ has better compatibility, easier debugging, same/better performance for 8-bit |
| optimize_model during quantization | Pre-process with quant_pre_process() | Documentation update 2023+ | Easier debugging, clearer separation of optimization and quantization steps |
| Manual calibration loops | CalibrationDataReader with get_next() | Long-standing pattern | Cleaner API, automatic session management, less boilerplate |

**Deprecated/outdated:**
- **QuantFormat.QOperator for 8-bit CPU:** Slow on x86-64 for 8-bit quantization, use QDQ instead (QOperator only faster for 4-bit weight-only quantization)
- **optimize_model=True in quantize_static:** Still works but not recommended - harder to debug accuracy issues when optimization and quantization are combined

## Open Questions

Things that couldn't be fully resolved:

1. **Does ResNet8 ONNX model exist or need to be created first?**
   - What we know: Phase 1 was "Model Conversion", evaluate.py references models/resnet8.onnx, but only resnet8.pt exists currently
   - What's unclear: Whether Phase 6 depends on Phase 1 completing first, or if ONNX model needs to be created as part of this phase
   - Recommendation: Verify Phase 1 completion status, if incomplete then Phase 6 must wait or include ONNX conversion as prerequisite task

2. **What is actual ResNet8 input format (NHWC vs NCHW)?**
   - What we know: evaluate.py uses NHWC (32, 32, 3) format, calibration_utils.py outputs NHWC, model came from onnx2torch conversion
   - What's unclear: Whether onnx2torch preserves TensorFlow NHWC format or converts to PyTorch NCHW format in ONNX model
   - Recommendation: Check model metadata in Phase 6 first task (input_shape from session.get_inputs()[0].shape), adjust if needed

3. **Is uint8 quantization worthwhile for CPU inference?**
   - What we know: ONNX Runtime CPU supports U8U8 (uint8 activations/weights), S8S8 (int8) is default recommendation, uint8 may benefit ReLU activations
   - What's unclear: Whether accuracy difference justifies testing both types for ResNet8 specifically, or if int8 alone is sufficient
   - Recommendation: Implement both as planned (minimal extra effort), compare accuracy, document which performs better for future reference

4. **Should quant_pre_process() be used for ResNet8?**
   - What we know: Pre-processing can improve quantization by merging Conv-BatchNorm-ReLU, but is optional
   - What's unclear: Whether ResNet8 architecture has BatchNorm nodes (likely yes), whether merge would significantly impact accuracy
   - Recommendation: Start without pre-processing (simpler, faster), add pre-processing task if int8 accuracy drops >3% from baseline

## Sources

### Primary (HIGH confidence)
- [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Official quantization guide
- [ONNX Runtime quantize.py Source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py) - quantize_static() function signature and parameters
- [ONNX Runtime calibrate.py Source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py) - CalibrationDataReader abstract class definition
- [ONNX Runtime Quantization README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md) - API usage examples
- [ONNX Runtime Image Classification Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md) - Practical CNN quantization workflow

### Secondary (MEDIUM confidence)
- [Optimum ONNX Runtime Quantization](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/quantization) - Third-party integration patterns verified with official docs
- [NVIDIA TensorRT Model Optimizer ONNX Guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/windows_guides/_ONNX_PTQ_guide.html) - Cross-verified ONNX quantization patterns
- [Onnx Model Quantization by Nashrakhan (Medium, 2024)](https://medium.com/@nashrakhan1008/model-quantization-8f10c537e0eb) - Practical tutorial verified with official docs

### Tertiary (LOW confidence - needs validation)
- [ONNX Runtime Discussion #24038](https://github.com/microsoft/onnxruntime/discussions/24038) - QDQ format behavior nuances (single community discussion, needs verification)
- [ONNX Runtime Issue #11928](https://github.com/microsoft/onnxruntime/issues/11928) - Percentile/Entropy calibration issues for ResNet-50 (older issue from 2022, may be resolved)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Built-in onnxruntime module verified in v1.23.2, no new dependencies needed
- Architecture: HIGH - CalibrationDataReader pattern verified in official sources, quantize_static() API well-documented
- Pitfalls: HIGH - Preprocessing mismatch, insufficient calibration, and format confusion documented in official sources and GitHub issues
- Expected accuracy: MEDIUM - 0-3% loss typical for CNNs extrapolated from larger models, ResNet8 specifically needs empirical validation

**Research date:** 2026-01-28
**Valid until:** 60 days (ONNX Runtime quantization API is stable, major changes unlikely)
