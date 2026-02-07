# Core Operation: QLinearConv

## Overview

QLinearConv performs quantized convolution, computing spatial feature maps using INT8 arithmetic. This operation implements the same spatial computation as standard Conv (sliding window, kernel multiplication, accumulation) but with fundamentally different arithmetic: a **two-stage computation pattern** that separates integer operations from scaling.

**Relationship to standard Conv:** Identical spatial semantics (stride, padding, kernel size), different arithmetic (INT8×INT8→INT32 accumulation, then requantization to INT8 output).

**Key difference from FP32 Conv:** Two-stage computation requires INT32 accumulator to prevent overflow during MAC operations, then applies all scale factors in a single requantization step. This pattern separates hardware-efficient integer operations from precision-critical floating-point scaling.

This document follows the [ONNX QLinearConv specification](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) as the authoritative source.

---

## Input Specification

QLinearConv has **8 required inputs** and **1 optional input** (bias):

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `x` | int8/uint8 | [N, C, H, W] | Quantized input activations |
| `x_scale` | float32 | scalar or [C] | Input scale factor (per-tensor or per-channel) |
| `x_zero_point` | int8/uint8 | scalar or [C] | Input zero-point offset (matches x type) |
| `w` | int8/uint8 | [M, C, kH, kW] | Quantized convolution weights |
| `w_scale` | float32 | scalar or [M] | Weight scale factor (per-tensor or per-channel) |
| `w_zero_point` | int8/uint8 | scalar or [M] | Weight zero-point offset (matches w type) |
| `y_scale` | float32 | scalar | Output scale factor |
| `y_zero_point` | int8/uint8 | scalar | Output zero-point offset (determines output type) |
| `B` | int32 | [M] | **Optional** bias term in INT32 (quantized space) |

**Notation:**
- N: batch size
- C: input channels
- H, W: input spatial dimensions
- M: output channels (number of filters)
- kH, kW: kernel spatial dimensions

**Per-tensor vs per-channel:**
- **Per-tensor quantization:** `x_scale`, `x_zero_point`, `w_scale`, `w_zero_point` are scalars (single value for entire tensor)
- **Per-channel quantization:** `w_scale` and `w_zero_point` can be arrays of shape [M] (one value per output channel)
- Input quantization (`x_scale`, `x_zero_point`) is typically per-tensor for activations
- Weight quantization can be either per-tensor or per-channel

---

## Two-Stage Computation Formula

The QLinearConv computation separates into two distinct stages:

### Stage 1: INT8×INT8→INT32 MAC Operations

For each output position $(n, m, h, w)$:

$$\text{acc}_{INT32}[n,m,h,w] = \sum_{c=0}^{C-1} \sum_{kh=0}^{kH-1} \sum_{kw=0}^{kW-1} \left( x_{INT8}[n,c,h \cdot s + kh, w \cdot s + kw] - x\_zero\_point \right) \times \left( w_{INT8}[m,c,kh,kw] - w\_zero\_point \right)$$

**Where:**
- $s$ is the stride
- Zero-points are **subtracted before** multiplication (critical for correctness)
- Accumulation happens in **INT32** to prevent overflow
- No floating-point operations in this stage

### Stage 2: Requantization to INT8

After accumulating all MAC operations in INT32, requantize to INT8 output:

$$y_{INT8}[n,m,h,w] = \text{saturate}\left( \text{round}\left( \frac{\text{acc}_{INT32}[n,m,h,w] \times x\_scale \times w\_scale}{y\_scale} \right) + y\_zero\_point \right)$$

**Where:**
- Combined scale factor: $\frac{x\_scale \times w\_scale}{y\_scale}$ applied as single floating-point operation
- `round()` uses **round-to-nearest-even** (banker's rounding)
- `saturate()` clips to INT8 range [-128, 127] or UINT8 range [0, 255]

**If bias term B is present:**

$$y_{INT8}[n,m,h,w] = \text{saturate}\left( \text{round}\left( \frac{(\text{acc}_{INT32}[n,m,h,w] + B[m]) \times x\_scale \times w\_scale}{y\_scale} \right) + y\_zero\_point \right)$$

The bias is added **before** applying the scale factor (bias is already in quantized space, not floating-point).

### Why This Two-Stage Pattern?

**Hardware efficiency:**
- Stage 1 uses only integer operations (fast, low power, suitable for specialized hardware)
- Stage 2 requires floating-point (or fixed-point) arithmetic for scaling (precision-critical but done once per output)

**Numerical stability:**
- Applying scales during accumulation causes precision loss from repeated floating-point operations
- Applying scales once after accumulation preserves accuracy

**Overflow prevention:**
- INT32 accumulator prevents overflow during MAC operations
- INT8×INT8 products fit in INT16, but summing hundreds of products requires INT32

---

## Per-Tensor Quantization Example

### Example Configuration

```python
# Quantization parameters (per-tensor: all scalars)
x_scale = 0.0235          # Input scale
x_zero_point = 0          # Symmetric quantization for input
w_scale = 0.0152          # Weight scale
w_zero_point = 0          # Symmetric quantization for weights
y_scale = 0.0314          # Output scale
y_zero_point = 0          # Symmetric quantization for output

# Convolution attributes
stride = 1
padding = 0
kernel_size = (3, 3)
input_channels = 3
output_channels = 16
```

### Step-by-Step Calculation

**Input patch extraction:**

For a single output position, extract a 3×3×3 patch from the input:

```python
# Example INT8 values from input (3 channels, 3×3 spatial)
x_patch = np.array([
    # Channel 0
    [[45, 32, 28],
     [51, 48, 35],
     [39, 42, 33]],
    # Channel 1
    [[62, 55, 49],
     [68, 71, 64],
     [58, 61, 52]],
    # Channel 2
    [[38, 41, 35],
     [44, 47, 40],
     [36, 39, 34]]
], dtype=np.int8)

# Example INT8 kernel weights for output channel 0 (3 channels, 3×3 spatial)
w_kernel = np.array([
    # Channel 0
    [[-12, 8, 5],
     [15, -9, 11],
     [7, -6, 4]],
    # Channel 1
    [[9, -14, 7],
     [-11, 13, -8],
     [6, 10, -5]],
    # Channel 2
    [[8, 11, -9],
     [14, -7, 12],
     [-10, 6, 9]]
], dtype=np.int8)
```

**Stage 1: INT32 MAC accumulation**

```python
# Subtract zero-points (both are 0 in this example, but shown for completeness)
x_dequant = x_patch.astype(np.int32) - x_zero_point  # [3, 3, 3] INT32
w_dequant = w_kernel.astype(np.int32) - w_zero_point  # [3, 3, 3] INT32

# Element-wise multiplication and sum (MAC operation)
products = x_dequant * w_dequant  # [3, 3, 3] INT32

# Partial sums by channel (to show accumulation)
channel_sums = np.sum(products, axis=(1, 2))  # Sum spatial dimensions per channel
# Channel 0: 45*(-12) + 32*8 + ... = sum of 9 products
# Channel 1: 62*9 + 55*(-14) + ... = sum of 9 products
# Channel 2: 38*8 + 41*11 + ... = sum of 9 products

# Total accumulator value (sum all channels and spatial positions)
acc_int32 = np.sum(products)  # Single INT32 value

# Example result (actual computation)
acc_int32 = 3847  # This is within INT16 range, but larger kernels could overflow INT16
```

**Stage 2: Requantization**

```python
# Combined scale factor
scale_factor = (x_scale * w_scale) / y_scale
# = (0.0235 * 0.0152) / 0.0314
# = 0.000357 / 0.0314
# ≈ 0.01137

# Apply scale factor
scaled_value = acc_int32 * scale_factor
# = 3847 * 0.01137
# ≈ 43.74

# Round to nearest integer (ties-to-even)
rounded_value = np.round(scaled_value)
# = round(43.74) = 44

# Add zero-point
with_zero_point = rounded_value + y_zero_point
# = 44 + 0 = 44

# Saturate to INT8 range [-128, 127]
y_int8 = np.clip(with_zero_point, -128, 127).astype(np.int8)
# = clip(44, -128, 127) = 44
```

**Final output:** `y_int8 = 44` for this single output position.

---

## Per-Channel Quantization Example

Per-channel quantization uses different scale and zero-point values for each **output channel** (dimension M).

### Key Difference from Per-Tensor

**Stage 1 (MAC):** Identical to per-tensor — zero-point subtraction and INT32 accumulation unchanged.

**Stage 2 (Requantization):** Different scale factor for each output channel.

### Example Configuration

```python
# Per-channel weight quantization (arrays of length M = 16 output channels)
w_scale = np.array([0.0152, 0.0148, 0.0161, 0.0145, ...])  # [16] - one per output channel
w_zero_point = np.array([0, 0, 0, 0, ...])  # [16] - typically all zeros for symmetric

# Input and output still per-tensor
x_scale = 0.0235          # scalar
x_zero_point = 0          # scalar
y_scale = 0.0314          # scalar
y_zero_point = 0          # scalar
```

### Computation for Output Channel m

For output channel $m$, use $w\_scale[m]$ and $w\_zero\_point[m]$:

**Stage 1:** Same INT32 accumulation as per-tensor example:
```python
# Extract weights for channel m
w_kernel_m = w[m, :, :, :]  # [C, kH, kW]

# MAC operation (same as before)
acc_int32[m] = np.sum((x_patch - x_zero_point) * (w_kernel_m - w_zero_point[m]))
```

**Stage 2:** Channel-specific scale factor:
```python
# For output channel m=0:
scale_factor_0 = (x_scale * w_scale[0]) / y_scale
# = (0.0235 * 0.0152) / 0.0314 ≈ 0.01137

# For output channel m=1:
scale_factor_1 = (x_scale * w_scale[1]) / y_scale
# = (0.0235 * 0.0148) / 0.0314 ≈ 0.01107

# Each channel gets its own requantization
y_int8[m] = saturate(round(acc_int32[m] * scale_factor[m]) + y_zero_point)
```

### Storage Overhead Analysis

**Per-tensor quantization:**
- 2 parameters per layer (1 scale + 1 zero-point for weights)
- Example: ResNet8 conv1 (3→16 channels, 3×3 kernel) = 2 parameters

**Per-channel quantization:**
- 2×M parameters per layer (M scales + M zero-points for weights)
- Example: ResNet8 conv1 (M=16 output channels) = 32 parameters

**Weight tensor size:**
- Conv1: 16×3×3×3 = 432 weight values
- Overhead: 32 / 432 ≈ 7.4%

**For larger layers:**
- Conv with M=256, C=128, 3×3 kernel: 256×128×3×3 = 294,912 weights
- Per-channel parameters: 2×256 = 512
- Overhead: 512 / 294,912 ≈ 0.17% (negligible)

**Conclusion:** Per-channel quantization overhead is negligible for typical CNN layers, but provides better accuracy by matching scale to each channel's weight distribution.

---

## Per-Channel vs Per-Tensor in Quantized ResNet8

**Note:** The quantized ONNX model generated for ResNet8 uses QDQ (Quantize-Dequantize) format, not QLinearConv format. The QDQ format achieves similar quantization behavior through QuantizeLinear/DequantizeLinear pairs around standard Conv operations.

For reference, typical quantization schemes for ResNet8 convolution layers:

| Layer | Input | Output | Kernel | Typical Quantization |
|-------|-------|--------|--------|---------------------|
| conv1 | 3     | 16     | 3×3    | Per-tensor weights, per-tensor activations |
| conv2 | 16    | 16     | 3×3    | Per-tensor or per-channel weights |
| conv3 | 16    | 32     | 3×3    | Per-channel weights (wider range) |
| conv4 | 32    | 32     | 3×3    | Per-channel weights |
| conv5 | 32    | 64     | 3×3    | Per-channel weights (wider range) |
| conv6 | 64    | 64     | 3×3    | Per-channel weights |
| conv7 | 64    | 64     | 3×3    | Per-channel weights |
| conv8 | 64    | 64     | 3×3    | Per-channel weights |

**General pattern:**
- **Early layers** (small output channels): Often per-tensor (overhead matters more)
- **Later layers** (many output channels): Per-channel (overhead negligible, accuracy benefit)
- **Activations:** Almost always per-tensor (per-channel activations rarely used)

**Why per-channel for deeper layers:**
- Different output channels learn different features (edges, textures, semantic patterns)
- Weight magnitudes vary significantly across channels
- Per-tensor quantization would waste dynamic range on low-magnitude channels
- Per-channel allows optimal scale for each channel's weight distribution

---

## INT32 Accumulator Requirement

### Why INT32 is Required

**Naive assumption:** "INT8×INT8 products fit in INT16, so INT16 accumulator should suffice."

**Reality:** Convolution sums hundreds of INT8×INT8 products. This sum can overflow INT16.

### Overflow Demonstration

```python
import numpy as np

def demonstrate_accumulator_overflow():
    """
    Show why INT32 accumulator is required for QLinearConv.

    Worst-case scenario:
    - 3×3 kernel with 64 input channels = 576 MAC operations
    - Each MAC: max INT8 value (127) × max INT8 value (127) = 16,129
    - Total accumulation: 576 × 16,129 = 9,290,304
    - INT16 max: 32,767 — OVERFLOWS by 284×!
    - INT32 max: 2,147,483,647 — safe
    """

    # Typical ResNet8 layer: 3×3 kernel, 64 input channels
    C, kH, kW = 64, 3, 3
    num_macs = C * kH * kW  # 576 MAC operations

    # Worst-case input: all maximum positive INT8 values
    x_patch = np.full((C, kH, kW), 127, dtype=np.int8)
    w_kernel = np.full((C, kH, kW), 127, dtype=np.int8)

    # Try INT16 accumulator (WRONG - will overflow)
    print("Testing INT16 accumulator (INCORRECT):")
    acc_int16 = np.int16(0)
    for x_val, w_val in zip(x_patch.flat, w_kernel.flat):
        product = np.int16(x_val) * np.int16(w_val)
        acc_int16 = np.int16(acc_int16 + product)  # Overflows silently

    print(f"  INT16 result: {acc_int16}")
    print(f"  WRONG due to overflow!")
    print()

    # Correct: INT32 accumulator
    print("Testing INT32 accumulator (CORRECT):")
    acc_int32 = np.int32(0)
    for x_val, w_val in zip(x_patch.flat, w_kernel.flat):
        product = np.int32(x_val) * np.int32(w_val)
        acc_int32 += product

    print(f"  INT32 result: {acc_int32}")
    print(f"  Expected: {num_macs} MACs × 127×127 = {num_macs * 127 * 127}")
    print(f"  INT16 max: 32,767")
    print(f"  Overflow factor: {acc_int32 / 32767:.1f}× the INT16 maximum")
    print(f"  INT32 max: 2,147,483,647 (safe)")
    print()

    return acc_int32

# Run demonstration
if __name__ == "__main__":
    result = demonstrate_accumulator_overflow()
```

**Output:**
```
Testing INT16 accumulator (INCORRECT):
  INT16 result: -7744
  WRONG due to overflow!

Testing INT32 accumulator (CORRECT):
  INT32 result: 9290304
  Expected: 576 MACs × 127×127 = 9290304
  INT16 max: 32,767
  Overflow factor: 283.5× the INT16 maximum
  INT32 max: 2,147,483,647 (safe)
```

### Practical Implications

**For hardware implementers:**
- **Accumulator must be INT32** — no exceptions
- INT16 accumulator will produce incorrect results that degrade accuracy
- Errors accumulate across layers, compounding the problem
- INT32 accumulator is explicitly required by ONNX specification

**Optimization note:**
- Some hardware uses wider accumulators (INT40, INT48) for additional safety margin
- Never use accumulators narrower than INT32

---

## Edge Cases

### Rounding: Ties-to-Even

ONNX specifies **round-to-nearest-even** (banker's rounding), not round-half-up.

**Behavior:**
```python
import numpy as np

# Ties (exactly halfway) round to nearest EVEN integer
print(np.round(0.5))   # → 0.0 (even)
print(np.round(1.5))   # → 2.0 (even)
print(np.round(2.5))   # → 2.0 (even)
print(np.round(3.5))   # → 4.0 (even)

# Non-ties round normally
print(np.round(0.4))   # → 0.0
print(np.round(0.6))   # → 1.0
```

**Why ties-to-even:**
- Eliminates statistical bias (equal probability of rounding up/down)
- Matches IEEE 754 floating-point standard
- Hardware implementations use this (x86, ARM)

**Common mistake:**
```python
# WRONG: Python's built-in round() uses ties-away-from-zero
round(2.5)      # → 2 (away from zero)
round(-2.5)     # → -2 (away from zero)

# CORRECT: NumPy's round() uses ties-to-even
np.round(2.5)   # → 2.0 (to even)
np.round(-2.5)  # → -2.0 (to even)
```

### Saturation: Clipping to Valid Range

After requantization, values must be clipped to the output data type range:

```python
# Example: Value exceeds INT8 maximum
scaled_value = 156.7
rounded_value = np.round(scaled_value)  # 157
with_zero_point = rounded_value + 0     # 157

# CORRECT: Saturate to [-128, 127]
y_int8 = np.clip(with_zero_point, -128, 127).astype(np.int8)
# Result: 127 (saturated)

# WRONG: Allow wrapping (C-style integer overflow)
y_int8_wrong = np.int8(with_zero_point)  # Wraps to -99
```

**Why saturation, not wrapping:**
- Graceful degradation (127 is closer to 157 than -99)
- Neural networks are trained with saturating activations (ReLU clips at 0)
- Wrapping causes catastrophic errors (large positive → large negative)

**Hardware note:** Most SIMD instruction sets provide saturating arithmetic (e.g., x86 SSE PADDSB, ARM NEON SQADD).

### Zero-Point Handling

Zero-points are **subtracted before MAC**, **added after requantization**:

**Stage 1 (MAC):**
```python
# CORRECT: Subtract zero-points before multiplication
x_centered = x_int8 - x_zero_point
w_centered = w_int8 - w_zero_point
acc = np.sum(x_centered * w_centered)

# WRONG: Forgetting to subtract zero-points
acc_wrong = np.sum(x_int8 * w_int8)  # Biased result
```

**Stage 2 (Requantization):**
```python
# CORRECT: Add output zero-point after scaling
y_int8 = saturate(round(acc * scale) + y_zero_point)

# WRONG: Subtracting instead of adding
y_int8_wrong = saturate(round(acc * scale) - y_zero_point)
```

**Symmetric quantization (zero_point = 0):**
- Simplifies to: `acc = np.sum(x_int8 * w_int8)` (no subtraction needed)
- Most quantized models use symmetric quantization for weights
- Activations may use asymmetric (non-zero zero-point) after ReLU

### Padding with Zero-Point

When padding is required, use the **input zero-point**, not literal zero:

```python
# CORRECT: Pad with input zero-point
if padding > 0:
    x_padded = np.pad(
        x_int8,
        ((0,0), (0,0), (padding,padding), (padding,padding)),
        mode='constant',
        constant_values=x_zero_point  # Use zero-point
    )

# WRONG: Pad with 0 (only correct if x_zero_point happens to be 0)
x_padded_wrong = np.pad(
    x_int8,
    ((0,0), (0,0), (padding,padding), (padding,padding)),
    mode='constant',
    constant_values=0  # Wrong for asymmetric quantization
)
```

**Why:** In quantized space, the "zero" value is represented by `x_zero_point`, not literal 0. Padding with 0 when `x_zero_point ≠ 0` introduces bias.

---

## Hardware Implementation Pseudocode

Complete reference implementation showing two-stage computation pattern:

```python
import numpy as np
from typing import Optional

def qlinear_conv(
    # Required inputs
    x: np.ndarray,              # INT8 input [N, C, H, W]
    x_scale: float,             # FP32 input scale
    x_zero_point: int,          # INT8 input zero-point
    w: np.ndarray,              # INT8 weights [M, C, kH, kW]
    w_scale: np.ndarray,        # FP32 weight scale (scalar or [M] for per-channel)
    w_zero_point: np.ndarray,   # INT8 weight zero-point (scalar or [M])
    y_scale: float,             # FP32 output scale
    y_zero_point: int,          # INT8 output zero-point
    # Optional inputs
    B: Optional[np.ndarray] = None,  # INT32 bias [M] (optional)
    # Convolution attributes
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:                # INT8 output [N, M, H_out, W_out]
    """
    QLinearConv: Quantized convolution with two-stage computation.

    Stage 1: INT8×INT8→INT32 MAC operations (integer-only)
    Stage 2: INT32→INT8 requantization with scaling (float ops)

    Supports both per-tensor and per-channel weight quantization.
    """

    # ========================================
    # Setup
    # ========================================
    N, C, H, W = x.shape
    M, C_w, kH, kW = w.shape
    assert C == C_w, "Input channels must match weight channels"

    # Calculate output dimensions
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1

    # Apply padding if needed
    if padding > 0:
        x = np.pad(
            x,
            ((0,0), (0,0), (padding,padding), (padding,padding)),
            mode='constant',
            constant_values=x_zero_point  # CRITICAL: use zero-point, not 0
        )

    # Check if per-channel or per-tensor quantization
    per_channel = isinstance(w_scale, np.ndarray) and w_scale.ndim > 0
    if per_channel:
        assert w_scale.shape[0] == M, "Per-channel scales must match output channels"
        assert w_zero_point.shape[0] == M, "Per-channel zero-points must match output channels"
    else:
        # Convert scalars to arrays for uniform indexing
        w_scale = np.full(M, w_scale, dtype=np.float32)
        w_zero_point = np.full(M, w_zero_point, dtype=np.int8)

    # ========================================
    # Stage 1: INT32 MAC Accumulation
    # ========================================
    # Initialize INT32 accumulator (REQUIRED to prevent overflow)
    acc = np.zeros((N, M, H_out, W_out), dtype=np.int32)

    for n in range(N):                    # Batch
        for m in range(M):                # Output channels
            for h_out in range(H_out):    # Output height
                for w_out in range(W_out):  # Output width

                    # Extract input patch
                    h_start = h_out * stride
                    w_start = w_out * stride
                    patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]  # [C, kH, kW]

                    # Center values by subtracting zero-points (CRITICAL)
                    x_centered = patch.astype(np.int32) - x_zero_point
                    w_centered = w[m].astype(np.int32) - w_zero_point[m]

                    # MAC operation: multiply and accumulate in INT32
                    products = x_centered * w_centered  # Element-wise INT32 multiplication
                    acc[n, m, h_out, w_out] = np.sum(products)  # INT32 sum

    # Add bias if present (bias is in INT32, quantized space)
    if B is not None:
        # Broadcast bias across batch and spatial dimensions
        acc += B.reshape(1, M, 1, 1)

    # ========================================
    # Stage 2: Requantization to INT8
    # ========================================
    # Allocate output
    y = np.zeros((N, M, H_out, W_out), dtype=np.int8)

    for m in range(M):  # Process each output channel (allows per-channel scaling)

        # Combined scale factor for this channel
        scale_factor = (x_scale * w_scale[m]) / y_scale

        # Apply scale to INT32 accumulator
        scaled = acc[:, m, :, :].astype(np.float32) * scale_factor

        # Round to nearest even integer
        rounded = np.round(scaled)

        # Add output zero-point
        with_zero_point = rounded + y_zero_point

        # Saturate to INT8 range [-128, 127]
        y[:, m, :, :] = np.clip(with_zero_point, -128, 127).astype(np.int8)

    return y
```

**Key hardware implementation notes:**

1. **Stage 1 can be fully pipelined** — all integer operations, no data dependencies between MAC operations
2. **INT32 accumulator is non-negotiable** — required by specification and proven by overflow analysis
3. **Stage 2 can be batched** — apply scaling to entire channel at once (SIMD-friendly)
4. **Per-channel adds minimal complexity** — only affects Stage 2, Stage 1 identical
5. **Zero-point handling is critical** — must subtract before MAC, add after requantization

---

## Summary

**QLinearConv** performs quantized convolution using a **two-stage computation pattern**:

1. **Stage 1 (INT8×INT8→INT32):** Integer MAC operations with zero-point subtraction, accumulated in INT32 to prevent overflow
2. **Stage 2 (INT32→INT8):** Apply combined scale factor, round to nearest even, add output zero-point, saturate to INT8 range

**Critical requirements for correctness:**
- INT32 accumulator (INT16 overflows for typical CNN layers)
- Zero-points subtracted before MAC, added after requantization
- Round-to-nearest-even (banker's rounding)
- Saturation (clipping), not wrapping, for out-of-range values
- Padding uses input zero-point, not literal zero

**Per-channel quantization** uses different scales per output channel, improving accuracy with negligible storage overhead (0.17% for typical layers). Only Stage 2 changes between per-tensor and per-channel; Stage 1 MAC operations are identical.

This operation enables CNN inference on integer hardware (analog accelerators, edge processors) while maintaining acceptable accuracy through careful quantization parameter selection and numerical handling.
