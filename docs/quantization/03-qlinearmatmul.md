# Core Operation: QLinearMatMul

## Overview

QLinearMatMul performs quantized matrix multiplication, computing output activations using INT8 arithmetic. This operation shares the same **two-stage computation pattern** as QLinearConv but without spatial dimensions — simpler sliding window logic, same fundamental arithmetic.

**Relationship to QLinearConv:** Identical two-stage pattern (INT8×INT8→INT32 MAC, then requantization), but operates on matrices instead of spatial feature maps. No stride, padding, or kernel extraction — just standard matrix multiplication with quantized values.

**Use case in CNNs:** Fully-connected (dense) layers. ResNet8 has one FC layer: final classification layer that maps 64 features to 10 classes (CIFAR-10).

This document follows the [ONNX QLinearMatMul specification](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) as the authoritative source.

---

## Input Specification

QLinearMatMul has **8 required inputs** (no bias term, unlike QLinearConv):

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `a` | int8/uint8 | [N, K] | Quantized input matrix (batch, features) |
| `a_scale` | float32 | scalar | Input scale factor |
| `a_zero_point` | int8/uint8 | scalar | Input zero-point offset (matches a type) |
| `b` | int8/uint8 | [K, M] | Quantized weight matrix (features, outputs) |
| `b_scale` | float32 | scalar or [M] | Weight scale factor (per-tensor or per-channel) |
| `b_zero_point` | int8/uint8 | scalar or [M] | Weight zero-point offset (matches b type) |
| `y_scale` | float32 | scalar | Output scale factor |
| `y_zero_point` | int8/uint8 | scalar | Output zero-point offset (determines output type) |

**Notation:**
- N: batch size
- K: input features (shared dimension for matrix multiplication)
- M: output features (classes, hidden units, etc.)

**Naming difference from QLinearConv:**
- QLinearConv uses `x`, `w` (input, weight)
- QLinearMatMul uses `a`, `b` (first matrix, second matrix)
- This follows standard ONNX MatMul naming convention

**No bias input:**
- QLinearConv has optional bias `B`
- QLinearMatMul does not support bias (add separately with QLinearAdd if needed)

**Per-channel quantization:**
- `b_scale` and `b_zero_point` can be arrays of shape [M] (one value per output feature)
- Input quantization (`a_scale`, `a_zero_point`) is always per-tensor (scalar)

---

## Two-Stage Computation Formula

For detailed explanation of the two-stage computation pattern, see [QLinearConv documentation](02-qlinearconv.md#two-stage-computation-formula).

QLinearMatMul uses the identical pattern — only the indexing changes (matrix indices instead of spatial):

### Stage 1: INT8×INT8→INT32 MAC Operations

For each output position $(n, m)$:

$$\text{acc}_{INT32}[n,m] = \sum_{k=0}^{K-1} \left( a_{INT8}[n,k] - a\_zero\_point \right) \times \left( b_{INT8}[k,m] - b\_zero\_point \right)$$

**Where:**
- $K$ is the shared dimension (input features)
- Zero-points are **subtracted before** multiplication (critical for correctness)
- Accumulation happens in **INT32** to prevent overflow
- No floating-point operations in this stage

### Stage 2: Requantization to INT8

After accumulating all MAC operations in INT32, requantize to INT8 output:

$$y_{INT8}[n,m] = \text{saturate}\left( \text{round}\left( \frac{\text{acc}_{INT32}[n,m] \times a\_scale \times b\_scale}{y\_scale} \right) + y\_zero\_point \right)$$

**Where:**
- Combined scale factor: $\frac{a\_scale \times b\_scale}{y\_scale}$ applied as single floating-point operation
- `round()` uses **round-to-nearest-even** (banker's rounding)
- `saturate()` clips to INT8 range [-128, 127] or UINT8 range [0, 255]

**For per-channel quantization:**

$$y_{INT8}[n,m] = \text{saturate}\left( \text{round}\left( \frac{\text{acc}_{INT32}[n,m] \times a\_scale \times b\_scale[m]}{y\_scale} \right) + y\_zero\_point \right)$$

Each output channel $m$ uses its own scale factor $b\_scale[m]$.

---

## Comparison to QLinearConv

The [QLinearConv operation](02-qlinearconv.md) shares the same fundamental computation pattern. The key differences are structural, not arithmetic:

| Aspect | QLinearConv | QLinearMatMul |
|--------|-------------|---------------|
| **Stage 1 pattern** | INT8×INT8→INT32 MAC | **Same** |
| **Stage 2 pattern** | Scale + requantize | **Same** |
| **Rounding** | Round-to-nearest-even | **Same** |
| **Saturation** | Clip to INT8 range | **Same** |
| **INT32 accumulator** | Required | **Required** |
| **Inputs** | 9 (with optional bias) | 8 (no bias) |
| **Spatial dimensions** | Yes (H, W) | No |
| **Stride, padding** | Yes | No |
| **Input names** | x, w | a, b |
| **Per-channel** | Common for weights | Less common (simpler layer) |

**Key insight:** If you understand QLinearConv, you understand QLinearMatMul. The only difference is **what values get multiplied** (spatial patches vs. matrix rows/columns) — the arithmetic is identical.

---

## ResNet8 FC Layer Example

ResNet8's final fully-connected layer performs classification:

**Layer configuration:**
- Input: 64 features (after global average pooling)
- Output: 10 classes (CIFAR-10)
- Weight matrix shape: [64, 10]
- Computation: Each of 10 outputs is dot product of 64-element input vector with 64-element weight column

**Quantization parameters (from ResNet8 INT8 model):**
```python
# Note: ResNet8 quantized model uses QDQ format (QuantizeLinear/DequantizeLinear),
# not QLinearMatMul. These values are illustrative based on typical quantization.

a_scale = 0.1903          # Input activations after average pooling
a_zero_point = 20         # Asymmetric (activations after ReLU)
b_scale = 0.0245          # FC layer weights (per-tensor for simplicity)
b_zero_point = 0          # Symmetric (typical for weights)
y_scale = 0.1585          # Output logits
y_zero_point = 0          # Symmetric (logits can be negative)
```

### Step-by-Step Calculation for One Output

**Computing output class 0 (first of 10 outputs):**

```python
import numpy as np

# Example INT8 input (64 features, batch size 1)
a_int8 = np.array([
    45, 32, 28, 51, 48, 35, 39, 42,
    62, 55, 49, 68, 71, 64, 58, 61,
    38, 41, 35, 44, 47, 40, 36, 39,
    52, 48, 44, 56, 59, 53, 50, 54,
    41, 38, 35, 43, 46, 40, 37, 41,
    55, 51, 48, 60, 63, 57, 54, 58,
    44, 40, 37, 47, 50, 44, 41, 45,
    58, 54, 51, 63, 66, 60, 57, 61
], dtype=np.int8)  # Shape: [64]

# Example INT8 weights for output class 0 (64 weights)
b_int8 = np.array([
    -12, 8, 5, 15, -9, 11, 7, -6,
    9, -14, 7, -11, 13, -8, 6, 10,
    8, 11, -9, 14, -7, 12, -10, 6,
    -15, 10, -8, 13, -11, 9, -7, 12,
    11, -9, 7, -13, 10, -8, 6, -11,
    14, -12, 9, -15, 13, -10, 8, -14,
    -10, 8, -6, 12, -9, 7, -5, 11,
    -13, 10, -8, 14, -11, 9, -7, 13
], dtype=np.int8)  # Shape: [64]
```

**Stage 1: INT32 MAC accumulation**

```python
# Subtract zero-points
a_centered = a_int8.astype(np.int32) - a_zero_point  # [64] INT32
b_centered = b_int8.astype(np.int32) - b_zero_point  # [64] INT32

# Example centered values (first 8 elements):
# a_centered: [25, 12, 8, 31, 28, 15, 19, 22, ...]
# b_centered: [-12, 8, 5, 15, -9, 11, 7, -6, ...] (unchanged, zero_point=0)

# Element-wise multiplication
products = a_centered * b_centered  # [64] INT32

# Example products (first 8 elements):
# [25*(-12), 12*8, 8*5, 31*15, 28*(-9), 15*11, 19*7, 22*(-6)]
# = [-300, 96, 40, 465, -252, 165, 133, -132]

# Accumulate (sum all 64 products)
acc_int32 = np.sum(products)
# Example result: 2847
```

**Stage 2: Requantization**

```python
# Combined scale factor
scale_factor = (a_scale * b_scale) / y_scale
# = (0.1903 * 0.0245) / 0.1585
# = 0.004662 / 0.1585
# ≈ 0.02941

# Apply scale factor
scaled_value = acc_int32 * scale_factor
# = 2847 * 0.02941
# ≈ 83.74

# Round to nearest even integer
rounded_value = np.round(scaled_value)
# = round(83.74) = 84

# Add zero-point
with_zero_point = rounded_value + y_zero_point
# = 84 + 0 = 84

# Saturate to INT8 range [-128, 127]
y_int8 = np.clip(with_zero_point, -128, 127).astype(np.int8)
# = clip(84, -128, 127) = 84
```

**Final output:** Class 0 logit = 84 (in quantized INT8 space)

Repeat this process for all 10 output classes. The class with the highest logit value is the predicted class.

---

## INT32 Accumulator for MatMul

See [INT32 accumulator demonstration](02-qlinearconv.md#int32-accumulator-requirement) for detailed proof of why INT32 is required.

**ResNet8 FC layer analysis:**

```python
# ResNet8 final layer: 64 input features × 1 output feature
K = 64  # Number of MAC operations per output

# Worst-case: all maximum positive INT8 values
max_product = 127 * 127  # 16,129
total_accumulation = max_product * K
# = 16,129 * 64
# = 1,032,256

# Compare to data type limits
INT16_MAX = 32,767
INT32_MAX = 2,147,483,647

overflow_factor = total_accumulation / INT16_MAX
# = 1,032,256 / 32,767
# ≈ 31.5×
```

**Conclusion:**
- ResNet8 FC layer: 64 MACs per output → accumulator up to 1,032,256
- INT16 max: 32,767 — **would overflow by 31.5×**
- INT32 max: 2,147,483,647 — **safe** (479× margin)

**Even this "small" fully-connected layer overflows INT16.** Larger FC layers (e.g., 512→1000 for ImageNet) have even more extreme overflow factors.

**INT32 accumulator is non-negotiable** for both QLinearConv and QLinearMatMul.

---

## Hardware Implementation Pseudocode

Complete reference implementation showing two-stage computation pattern:

```python
import numpy as np
from typing import Optional

def qlinear_matmul(
    # Required inputs
    a: np.ndarray,              # INT8 input [N, K]
    a_scale: float,             # FP32 input scale
    a_zero_point: int,          # INT8 input zero-point
    b: np.ndarray,              # INT8 weights [K, M]
    b_scale: np.ndarray,        # FP32 weight scale (scalar or [M] for per-channel)
    b_zero_point: np.ndarray,   # INT8 weight zero-point (scalar or [M])
    y_scale: float,             # FP32 output scale
    y_zero_point: int,          # INT8 output zero-point
) -> np.ndarray:                # INT8 output [N, M]
    """
    QLinearMatMul: Quantized matrix multiplication with two-stage computation.

    Stage 1: INT8×INT8→INT32 MAC operations (integer-only)
    Stage 2: INT32→INT8 requantization with scaling (float ops)

    Supports both per-tensor and per-channel weight quantization.
    """

    # ========================================
    # Setup
    # ========================================
    N, K = a.shape
    K_b, M = b.shape
    assert K == K_b, "Shared dimension K must match between a and b"

    # Check if per-channel or per-tensor quantization
    per_channel = isinstance(b_scale, np.ndarray) and b_scale.ndim > 0
    if per_channel:
        assert b_scale.shape[0] == M, "Per-channel scales must match output features"
        assert b_zero_point.shape[0] == M, "Per-channel zero-points must match output features"
    else:
        # Convert scalars to arrays for uniform indexing
        b_scale = np.full(M, b_scale, dtype=np.float32)
        b_zero_point = np.full(M, b_zero_point, dtype=np.int8)

    # ========================================
    # Stage 1: INT32 MAC Accumulation
    # ========================================
    # Initialize INT32 accumulator (REQUIRED to prevent overflow)
    acc = np.zeros((N, M), dtype=np.int32)

    for n in range(N):          # Batch
        for m in range(M):      # Output features
            for k in range(K):  # Shared dimension (dot product)

                # Center values by subtracting zero-points (CRITICAL)
                a_val = np.int32(a[n, k]) - a_zero_point
                b_val = np.int32(b[k, m]) - b_zero_point[m]

                # MAC operation: multiply and accumulate in INT32
                acc[n, m] += a_val * b_val

    # ========================================
    # Stage 2: Requantization to INT8
    # ========================================
    # Allocate output
    y = np.zeros((N, M), dtype=np.int8)

    for m in range(M):  # Process each output feature (allows per-channel scaling)

        # Combined scale factor for this output feature
        scale_factor = (a_scale * b_scale[m]) / y_scale

        # Apply scale to INT32 accumulator
        scaled = acc[:, m].astype(np.float32) * scale_factor

        # Round to nearest even integer
        rounded = np.round(scaled)

        # Add output zero-point
        with_zero_point = rounded + y_zero_point

        # Saturate to INT8 range [-128, 127]
        y[:, m] = np.clip(with_zero_point, -128, 127).astype(np.int8)

    return y
```

**Key hardware implementation notes:**

1. **Simpler than QLinearConv** — no spatial dimensions, stride, padding, or kernel extraction
2. **Stage 1 is standard matrix multiplication** — highly optimized on all hardware
3. **INT32 accumulator still required** — even small FC layers overflow INT16
4. **Stage 2 identical to QLinearConv** — same scaling, rounding, saturation logic
5. **Per-channel adds minimal complexity** — only affects Stage 2, Stage 1 unchanged

**Optimization opportunities:**
- Stage 1 can use optimized GEMM (General Matrix Multiply) kernels
- SIMD parallelization: compute multiple outputs simultaneously
- Batch processing: compute entire batch with single matrix operation

---

## Summary

**QLinearMatMul** performs quantized matrix multiplication using the same **two-stage computation pattern** as QLinearConv:

1. **Stage 1 (INT8×INT8→INT32):** Integer MAC operations with zero-point subtraction, accumulated in INT32 to prevent overflow
2. **Stage 2 (INT32→INT8):** Apply combined scale factor, round to nearest even, add output zero-point, saturate to INT8 range

**Relationship to QLinearConv:**
- **Same arithmetic pattern** — only structural differences (no spatial dimensions, stride, padding)
- **Same critical requirements** — INT32 accumulator, round-to-nearest-even, saturation
- **Simpler implementation** — standard matrix multiplication instead of convolution loops

**ResNet8 use case:**
- Final FC layer: 64 features → 10 classes
- 64 MACs per output
- Accumulator up to 1,032,256 (31.5× INT16 max) — proves INT32 requirement

**Per-channel quantization** uses different scales per output feature (column of weight matrix). Less common for FC layers than for convolutions, but follows identical pattern.

This operation enables efficient inference for fully-connected layers on integer hardware, completing the quantized operation set needed for CNN inference.
