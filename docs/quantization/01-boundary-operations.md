# Boundary Operations: QuantizeLinear and DequantizeLinear

## Overview

Boundary operations perform conversions between floating-point and quantized integer representations at model input/output boundaries. These operations implement the fundamental transformations needed for INT8 inference:

- **QuantizeLinear**: Converts FP32/FP16 tensors to INT8/UINT8 (model input quantization)
- **DequantizeLinear**: Converts INT8/UINT8 tensors back to FP32/FP16 (model output dequantization)

This document follows the [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) and [ONNX DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) specifications as authoritative sources.

---

## QuantizeLinear Operation

### Formula

The quantization operation follows:

$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

**Where:**
- Division uses **round-to-nearest-even** (banker's rounding)
- Saturation clips to the output data type range

### Parameters

**Inputs:**
- `x`: Input tensor to quantize (float32, float16, bfloat16, or int32)
- `y_scale`: Scale factor (scalar or tensor matching x dimensions)
- `y_zero_point`: Zero point offset (optional, defaults to 0 for the output type)

**Output:**
- `y`: Quantized tensor (int8, uint8, int16, uint16, int4, uint4, int2, uint2)
- Output type determined by `y_zero_point` data type

### How the Formula is Formed

The quantization formula maps continuous floating-point values to discrete integer values:

1. **Scale division** (`x / y_scale`): Normalizes the input to quantization step units. A smaller `y_scale` means finer quantization granularity.

2. **Rounding** (`round(...)`): Maps real values to nearest integers. ONNX uses round-to-nearest-even (banker's rounding) to avoid systematic bias:
   - 0.5 → 0 (rounds to even)
   - 1.5 → 2 (rounds to even)
   - 2.5 → 2 (rounds to even)
   - -0.5 → 0 (rounds to even)

3. **Zero-point shift** (`+ y_zero_point`): Offsets the quantized range to match the desired integer type. This allows asymmetric distributions to use the full integer range.

4. **Saturation** (`saturate(...)`): Clips values to the output type's representable range, preventing overflow.

### Saturation Ranges

The output is clipped to these ranges based on the output data type:

| Data Type | Range | Bits | Typical Use |
|-----------|-------|------|-------------|
| uint2 | [0, 3] | 2 | Experimental ultra-low precision |
| int2 | [-2, 1] | 2 | Experimental ultra-low precision |
| uint4 | [0, 15] | 4 | Sub-byte quantization |
| int4 | [-8, 7] | 4 | Sub-byte quantization |
| uint8 | [0, 255] | 8 | Standard activations (ReLU) |
| int8 | [-128, 127] | 8 | Standard weights/activations |
| uint16 | [0, 65535] | 16 | High-precision quantization |
| int16 | [-32768, 32767] | 16 | High-precision quantization |

### Symmetric Quantization (y_zero_point = 0)

When `y_zero_point = 0`, the formula simplifies to:

$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right)\right)$$

**Characteristics:**
- Zero in floating-point maps exactly to 0 in quantized space
- Symmetric range around zero: [-127, 127] for int8 (note: -128 typically unused)
- Simpler hardware implementation (no zero-point addition)
- Ideal for distributions centered near zero (e.g., post-BatchNorm activations)

**Example calculation:**
```
Given: x = 2.7, y_scale = 0.1, y_zero_point = 0, output type = int8

1. Divide: 2.7 / 0.1 = 27.0
2. Round: round(27.0) = 27
3. Add zero-point: 27 + 0 = 27
4. Saturate: 27 is within [-128, 127], no clipping
5. Result: y = 27
```

### Asymmetric Quantization (y_zero_point ≠ 0)

When `y_zero_point ≠ 0`, the full formula applies:

$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

**Characteristics:**
- Zero in floating-point can map to any quantized value
- Full range utilization: [-128, 127] for int8 or [0, 255] for uint8
- Better accuracy for asymmetric distributions (e.g., ReLU activations with range [0, 6])
- Requires zero-point addition in hardware (slightly more complex)

**Example calculation:**
```
Given: x = 2.7, y_scale = 0.1, y_zero_point = 10, output type = int8

1. Divide: 2.7 / 0.1 = 27.0
2. Round: round(27.0) = 27
3. Add zero-point: 27 + 10 = 37
4. Saturate: 37 is within [-128, 127], no clipping
5. Result: y = 37
```

### Numerical Example: Symmetric vs Asymmetric

**Scenario:** Quantize weights in range [-1.2, 0.8] to int8

**Symmetric approach (zero_point = 0):**
- Range: max(|-1.2|, |0.8|) = 1.2
- Scale: (2 × 1.2) / 255 ≈ 0.00941
- Effective range: [-1.2, 1.2] (wastes 0.4 of upper range)
- Example values:
  - -1.2 → saturate(round(-1.2 / 0.00941)) = saturate(-127.5) = -128
  - 0.0 → saturate(round(0.0 / 0.00941)) = 0
  - 0.8 → saturate(round(0.8 / 0.00941)) = saturate(85.0) = 85

**Asymmetric approach (zero_point ≠ 0):**
- Scale: (0.8 - (-1.2)) / 255 ≈ 0.00784
- Zero-point: round(-1.2 / 0.00784) + 0 = -153 → Adjust to map -1.2 to -128: zero_point = 25
- Effective range: [-1.2, 0.8] (full utilization)
- Example values:
  - -1.2 → saturate(round(-1.2 / 0.00784) + 25) = saturate(-128)
  - 0.0 → saturate(round(0.0 / 0.00784) + 25) = saturate(25) = 25
  - 0.8 → saturate(round(0.8 / 0.00784) + 25) = saturate(127) = 127

**Trade-off:** Asymmetric quantization uses the integer range more efficiently, providing better accuracy, but requires hardware to perform zero-point addition.

---

## DequantizeLinear Operation

### Formula

The dequantization operation follows:

$$y = (x - x\_zero\_point) \times x\_scale$$

**Note:** Parameter names use the `x_` prefix (not `y_`) because this operation takes quantized input and produces floating-point output.

### Parameters

**Inputs:**
- `x`: Quantized input tensor (int8, uint8, int16, uint16, int4, uint4, int2, uint2)
- `x_scale`: Scale factor (scalar or tensor matching x dimensions)
- `x_zero_point`: Zero point offset (optional, defaults to 0, must match x data type)

**Output:**
- `y`: Dequantized tensor (float32, float16, or bfloat16)

### How the Formula is Formed

The dequantization formula is the inverse of quantization, reconstructing continuous values from discrete integers:

1. **Zero-point shift** (`x - x_zero_point`): Removes the offset applied during quantization, centering the quantized values.

2. **Scale multiplication** (`... × x_scale`): Converts quantization step units back to the original floating-point range. This is the inverse of the division in QuantizeLinear.

### Numerical Example

Continuing from the QuantizeLinear symmetric example:

```
Given: x = 27 (quantized value), x_scale = 0.1, x_zero_point = 0

1. Subtract zero-point: 27 - 0 = 27
2. Multiply by scale: 27 × 0.1 = 2.7
3. Result: y = 2.7
```

For the asymmetric example:

```
Given: x = 37 (quantized value), x_scale = 0.1, x_zero_point = 10

1. Subtract zero-point: 37 - 10 = 27
2. Multiply by scale: 27 × 0.1 = 2.7
3. Result: y = 2.7
```

Both recover the original value `x = 2.7` exactly (within floating-point precision).

---

## Round-Trip Relationship

### Perfect Reconstruction Condition

For any input `x`, the composition of quantization and dequantization approximately recovers the original value:

$$\text{Dequant}(\text{Quant}(x)) \approx x$$

The reconstruction is exact when:
- The original value `x` falls exactly on a quantization step (i.e., `x / y_scale` is an integer)
- No saturation occurs

### Error Bound

For values **within the quantization range** (no saturation), the maximum round-trip error is bounded by:

$$\left|x - \text{Dequant}(\text{Quant}(x))\right| \leq \frac{y\_scale}{2}$$

**Proof:**

The error comes from rounding in the quantization step:

1. Let $q = \frac{x}{y\_scale}$ be the exact (unrounded) quantized value
2. After rounding: $\text{round}(q)$ has error $|q - \text{round}(q)| \leq 0.5$ (maximum rounding error)
3. After dequantization: $\text{Dequant}(\text{Quant}(x)) = \text{round}(q) \times y\_scale$
4. Error: $|x - \text{Dequant}(\text{Quant}(x))| = |q \times y\_scale - \text{round}(q) \times y\_scale| = |q - \text{round}(q)| \times y\_scale \leq 0.5 \times y\_scale$

**Example:**
- If `y_scale = 0.01`, maximum error is 0.005 (half a percent)
- If `y_scale = 0.1`, maximum error is 0.05

### Saturation Behavior

For values **outside the quantization range**, saturation introduces additional error:

- If $x < x_{\min}$ (below quantization range): $\text{error} = |x - x_{\min}|$
- If $x > x_{\max}$ (above quantization range): $\text{error} = |x - x_{\max}|$

Where $x_{\min}$ and $x_{\max}$ are the minimum and maximum representable values after dequantization:

$$x_{\min} = (q_{\min} - y\_zero\_point) \times y\_scale$$
$$x_{\max} = (q_{\max} - y\_zero\_point) \times y\_scale$$

And $q_{\min}$, $q_{\max}$ are the saturation bounds (e.g., -128, 127 for int8).

### Numerical Example: Round-Trip Error

**Case 1: Value within range**

```
Given: x = 2.73, y_scale = 0.1, y_zero_point = 0, output type = int8

Quantize:
1. Divide: 2.73 / 0.1 = 27.3
2. Round: round(27.3) = 27
3. Add zero-point: 27 + 0 = 27
4. Saturate: 27 (no clipping needed)
Result: y = 27

Dequantize:
1. Subtract zero-point: 27 - 0 = 27
2. Multiply: 27 × 0.1 = 2.7
Result: y = 2.7

Error: |2.73 - 2.7| = 0.03
Bound check: 0.03 ≤ 0.1/2 = 0.05 ✓
```

**Case 2: Saturation (out of range)**

```
Given: x = 15.0, y_scale = 0.1, y_zero_point = 0, output type = int8

Quantize:
1. Divide: 15.0 / 0.1 = 150.0
2. Round: round(150.0) = 150
3. Add zero-point: 150 + 0 = 150
4. Saturate: 150 > 127, clips to 127
Result: y = 127

Dequantize:
1. Subtract zero-point: 127 - 0 = 127
2. Multiply: 127 × 0.1 = 12.7
Result: y = 12.7

Error: |15.0 - 12.7| = 2.3
This exceeds the bound (0.05) because saturation occurred.
```

---

## Data Type Reference

Complete table of ONNX-supported quantization types:

| Data Type | Range | Bits | Storage | Common Use Case |
|-----------|-------|------|---------|-----------------|
| uint2 | [0, 3] | 2 | Packed | Experimental ultra-low precision |
| int2 | [-2, 1] | 2 | Packed | Experimental ultra-low precision |
| uint4 | [0, 15] | 4 | Packed | Weight-only quantization |
| int4 | [-8, 7] | 4 | Packed | Weight-only quantization |
| uint8 | [0, 255] | 8 | 1 byte | Activations (ReLU, always positive) |
| int8 | [-128, 127] | 8 | 1 byte | Weights and general activations |
| uint16 | [0, 65535] | 16 | 2 bytes | High-precision quantization |
| int16 | [-32768, 32767] | 16 | 2 bytes | Accumulator intermediate values |

**Note:** The output data type of QuantizeLinear is determined by the data type of the `y_zero_point` parameter. For symmetric quantization (zero_point = 0), int8 is typical.

---

## Summary

**QuantizeLinear** converts floating-point tensors to quantized integers using scale division, rounding, zero-point offset, and saturation. The operation supports both symmetric (simpler, zero-centered) and asymmetric (better range utilization) quantization schemes.

**DequantizeLinear** performs the inverse transformation, reconstructing floating-point values from quantized integers by removing the zero-point offset and multiplying by the scale factor.

**Round-trip error** is bounded by half the quantization step size (±y_scale/2) for values within range, with larger errors occurring only when saturation clips out-of-range values.

These boundary operations enable neural network inference on integer hardware while maintaining acceptable accuracy, with the scale and zero-point parameters controlling the trade-off between representable range and precision.
