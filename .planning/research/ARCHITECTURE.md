# Architecture Research: Quantized Operations Flow

**Project:** ResNet8 Quantized Operations Documentation
**Milestone:** v1.3 - Hardware Accelerator Implementation Reference
**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

Quantized inference transforms the compute graph from floating-point operations to integer arithmetic with explicit scale/zero-point parameters. Two primary formats exist: **QDQ format** (Quantize-DeQuantize pairs around FP32 ops) and **Pure Integer format** (QLinear ops with embedded parameters). ResNet8's quantized graph uses QDQ format, where data flows as: FP32 input → QuantizeLinear → integer ops → DequantizeLinear → FP32 output. Residual connections require careful scale matching at Add operations. This document provides the operation-level details needed for hardware accelerator implementation.

## What This Document Provides

This architecture research focuses on **quantized operations flow** for hardware accelerator implementation:

1. **Graph structure**: Where QuantizeLinear/DequantizeLinear appear in the computation graph
2. **Data flow**: How FP32 tensors convert to int8, flow through operations, and convert back
3. **Operation internals**: Scale/zero-point parameters and integer arithmetic formulas
4. **Residual connections**: How skip connections work with quantization scale matching
5. **Format comparison**: QDQ vs pure integer graph structures

**NOT covered** (see existing ARCHITECTURE.md for PTQ integration):
- How to build quantization scripts
- Calibration data preparation
- Framework-specific quantization APIs
- Build order and phase planning

## Quantized Operations Overview

### The Quantization Formula

**Quantization (FP32 → int8):**
```
y_quantized = saturate(round(x_float / scale) + zero_point)
```

**Dequantization (int8 → FP32):**
```
y_float = (x_quantized - zero_point) * scale
```

**Parameters:**
- `scale`: Floating-point scaling factor (determines quantization granularity)
- `zero_point`: Integer offset (enables asymmetric quantization ranges)
- `saturate`: Clips to valid integer range (int8: [-128, 127], uint8: [0, 255])
- `round`: Rounds to nearest even (banker's rounding)

### Scale and Zero-Point Meaning

**Scale** determines the quantization step size:
- Smaller scale = finer quantization (better precision, smaller range)
- Larger scale = coarser quantization (wider range, less precision)
- Calculated during calibration: `scale = (max - min) / (2^bits - 1)`

**Zero-point** maps the zero value in floating-point to an integer:
- For int8 symmetric quantization: `zero_point = 0`
- For uint8 asymmetric quantization: `zero_point = round(-min / scale)`
- Ensures zero in FP32 maps exactly to an integer (important for ReLU)

## QDQ Format vs Pure Integer Format

### QDQ Format (QuantizeLinear-DequantizeLinear)

**Structure:** Original FP32 operators remain, wrapped with Q/DQ pairs.

```
Input (FP32)
    |
    v
QuantizeLinear (FP32 → int8)
    | (int8 tensor)
    v
DequantizeLinear (int8 → FP32)
    |
    v
Conv (FP32 operator, unchanged)
    |
    v
QuantizeLinear (FP32 → int8)
    | (int8 tensor)
    v
DequantizeLinear (int8 → FP32)
    |
    v
Output (FP32)
```

**Characteristics:**
- Original operators (Conv, MatMul, Add) are FP32 in the graph
- Runtime fuses Q/DQ pairs with operators for integer execution
- Graph remains readable and debuggable
- ONNX Runtime default format
- PyTorch quantization produces similar structure

**Advantages:**
- Flexible: Many operators quantized without needing Q-operator variants
- Debuggable: Can inspect intermediate quantized/dequantized values
- Framework-friendly: Easier to optimize automatically

**Disadvantages:**
- Runtime must perform fusion optimization
- Graph appears to be FP32 (misleading without inspection)

### Pure Integer Format (QLinear Operators)

**Structure:** Operators explicitly replaced with quantized variants.

```
Input (FP32)
    |
    v
QuantizeLinear (FP32 → int8)
    | (int8 tensor)
    v
QLinearConv (int8 → int8 directly)
    | (int8 tensor)
    v
QLinearMatMul (int8 → int8 directly)
    | (int8 tensor)
    v
DequantizeLinear (int8 → FP32)
    |
    v
Output (FP32)
```

**Characteristics:**
- Operators explicitly quantized (QLinearConv, QLinearMatMul, etc.)
- Each operator carries scale/zero-point parameters as inputs
- No intermediate dequantization between quantized ops
- ONNX Runtime QOperator format

**Advantages:**
- Explicit integer operations in graph
- Clear what hardware accelerator must implement
- No fusion required

**Disadvantages:**
- Limited operator coverage (not all ops have QLinear variants)
- Less flexible for uncommon operations
- ONNX Runtime warns: "S8S8 with QOperator will be slow on x86-64 CPUs"

### Format Comparison Table

| Aspect | QDQ Format | Pure Integer Format |
|--------|------------|---------------------|
| **Operators** | FP32 ops (Conv, MatMul) | Q-ops (QLinearConv, QLinearMatMul) |
| **Intermediate representation** | int8 wrapped as FP32 | Native int8 |
| **Runtime optimization** | Requires fusion | Direct execution |
| **Graph readability** | Original structure preserved | Explicit quantization |
| **Hardware mapping** | Runtime-dependent | Direct operator mapping |
| **ONNX Runtime recommendation** | QDQ (default) | QOperator avoided on x86 |
| **PyTorch approach** | Similar to QDQ | Not typically used |

## ResNet8 Quantized Graph Structure

### FP32 Baseline Graph

```
Input: (1, 32, 32, 3) FP32, range [0, 255]
    |
    v
[Initial Block]
Conv2D(16, 3x3) → BatchNorm → ReLU
    |
    v
[Stack 1: 16 filters]
├─ Residual Block 1
│  ├─ Conv2D(16, 3x3) → BN → ReLU
│  ├─ Conv2D(16, 3x3) → BN
│  └─ Add (with identity) → ReLU
│
├─ Residual Block 2
│  ├─ Conv2D(16, 3x3) → BN → ReLU
│  ├─ Conv2D(16, 3x3) → BN
│  └─ Add (with identity) → ReLU
    |
    v
[Stack 2: 32 filters, stride=2]
├─ Residual Block 3
│  ├─ Conv2D(32, 3x3, s=2) → BN → ReLU
│  ├─ Conv2D(32, 3x3) → BN
│  ├─ Shortcut: Conv2D(32, 1x1, s=2) → BN
│  └─ Add (with shortcut) → ReLU
│
├─ Residual Block 4
│  ├─ Conv2D(32, 3x3) → BN → ReLU
│  ├─ Conv2D(32, 3x3) → BN
│  └─ Add (with identity) → ReLU
    |
    v
[Stack 3: 64 filters, stride=2]
├─ Residual Block 5
│  ├─ Conv2D(64, 3x3, s=2) → BN → ReLU
│  ├─ Conv2D(64, 3x3) → BN
│  ├─ Shortcut: Conv2D(64, 1x1, s=2) → BN
│  └─ Add (with shortcut) → ReLU
│
├─ Residual Block 6
│  ├─ Conv2D(64, 3x3) → BN → ReLU
│  ├─ Conv2D(64, 3x3) → BN
│  └─ Add (with identity) → ReLU
    |
    v
[Classification Head]
GlobalAveragePooling → Dense(10) → Softmax
    |
    v
Output: (1, 10) FP32 probabilities
```

### QDQ Quantized Graph (ONNX Runtime)

**Key transformation:** Insert Q/DQ pairs at strategic points.

```
Input: (1, 32, 32, 3) FP32 [0, 255]
    |
    v
┌──────────────────────────────────┐
│ QuantizeLinear                   │  ← INPUT QUANTIZATION
│   scale: 1.0                     │     (input already scaled for uint8)
│   zero_point: 0                  │
└──────────────────────────────────┘
    | int8/uint8 tensor
    v
┌──────────────────────────────────┐
│ DequantizeLinear                 │
└──────────────────────────────────┘
    | FP32 (for computation)
    v
[Initial Conv - QDQ wrapped]
    ┌─────────────────────────┐
    │ Conv2D(16, 3x3)         │  ← Original FP32 operator
    │   Weights: quantized    │     Runtime fuses with Q/DQ
    └─────────────────────────┘
    | FP32
    v
┌──────────────────────────────────┐
│ QuantizeLinear                   │  ← ACTIVATION QUANTIZATION
│   scale: s1 (calibrated)         │     (learned during calibration)
│   zero_point: z1                 │
└──────────────────────────────────┘
    | int8 tensor
    v
┌──────────────────────────────────┐
│ DequantizeLinear                 │
└──────────────────────────────────┘
    | FP32
    v
BatchNorm (fused or separate) → ReLU
    |
    v
[Residual Block - Detailed structure]

    Main path (left):
        | FP32
        v
    QuantizeLinear (scale: s2, zp: z2)
        | int8
        v
    DequantizeLinear
        | FP32
        v
    Conv2D(16, 3x3)
        | FP32
        v
    QuantizeLinear (scale: s3, zp: z3)
        | int8
        v
    DequantizeLinear
        | FP32
        v
    BatchNorm → ReLU
        | FP32
        v
    QuantizeLinear (scale: s4, zp: z4)
        | int8
        v
    DequantizeLinear
        | FP32
        v
    Conv2D(16, 3x3)
        | FP32
        v
    QuantizeLinear (scale: s5, zp: z5)
        | int8
        v
    DequantizeLinear
        | FP32
        v
    BatchNorm
        | FP32
        |
        ├───────────────┐
        |               |
        v               v
    [To Add]      [Identity path]
                      |
                      v
                  QuantizeLinear (scale: s6, zp: z6)
                      | int8
                      v
                  DequantizeLinear
                      | FP32
                      |
                      v
                  [To Add]

    Add operation:  ← CRITICAL: Scale matching
        | Both inputs FP32, same scale
        v
    Add (element-wise)
        | FP32
        v
    ReLU
        | FP32
        v
    QuantizeLinear (scale: s7, zp: z7)
        | int8
        v
    [Continue to next block...]
```

**Key observations:**

1. **Input boundary:** Single Q/DQ pair converts FP32 input to int8
2. **Weight quantization:** Conv weights stored as int8 with scale/zero-point
3. **Activation quantization:** After each activation function (ReLU), insert Q/DQ
4. **Residual paths:** BOTH main and skip paths quantized before Add
5. **Output boundary:** Final DQ converts int8 back to FP32 for classification head

### Pure Integer Variant (QOperator format)

```
Input: (1, 32, 32, 3) FP32
    |
    v
┌──────────────────────────────────────────────────────┐
│ QuantizeLinear                                       │
│   scale: 1.0, zero_point: 0                          │
└──────────────────────────────────────────────────────┘
    | uint8 tensor
    v
┌──────────────────────────────────────────────────────┐
│ QLinearConv                                          │  ← Integer convolution
│   Inputs:                                            │
│     - x: uint8 tensor                                │
│     - x_scale: 1.0, x_zero_point: 0                  │
│     - w: int8 weights                                │
│     - w_scale: per-channel scales                    │
│     - w_zero_point: 0 (symmetric)                    │
│     - y_scale: s1, y_zero_point: z1                  │
│   Output: uint8 tensor (directly)                    │
└──────────────────────────────────────────────────────┘
    | uint8 (no intermediate dequantization)
    v
[Subsequent QLinearConv operations...]
    | uint8
    v
┌──────────────────────────────────────────────────────┐
│ DequantizeLinear                                     │
│   scale: s_final, zero_point: z_final                │
└──────────────────────────────────────────────────────┘
    | FP32
    v
Output
```

**Differences from QDQ:**
- No intermediate DequantizeLinear between operations
- Each QLinearConv explicitly carries 6-8 inputs (data + scale/zp for input/weight/output)
- Integer tensors flow directly between quantized operators
- Hardware accelerator sees explicit QLinearConv operations

## Data Flow Through Quantized ResNet8

### Complete Data Flow (QDQ Format)

**Stage 1: Input Quantization**

```
Input: Raw CIFAR-10 image
  Type: FP32
  Shape: (1, 32, 32, 3)
  Range: [0, 255]

    ↓ QuantizeLinear(scale=1.0, zp=0)

Quantized Input:
  Type: uint8
  Shape: (1, 32, 32, 3)
  Range: [0, 255]
  Interpretation: Each pixel value unchanged numerically
```

**Stage 2: Initial Convolution**

```
Quantized Input (uint8)
    ↓ DequantizeLinear
FP32 tensor [0, 255]
    ↓ Conv2D (FP32 operator, but optimized for quantized exec)
       Weights: int8 with scale/zp
       Computation: (x_fp32 * w_dequantized) + bias
FP32 activations
    ↓ QuantizeLinear(scale=s_conv1, zp=z_conv1)
int8 tensor
```

**Hardware accelerator perspective:**
```
Integer convolution (fused operation):
1. Input: uint8 pixels
2. Weights: int8 with scale w_scale
3. Computation: Σ(x_u8 * w_i8) → int32 accumulator
4. Scale: int32_acc * (input_scale * w_scale / output_scale)
5. Add zero-point: result + output_zp
6. Saturate: clip to int8 range [-128, 127]
7. Output: int8 activations
```

**Stage 3: Residual Block (Main Path)**

```
Input to residual block: int8 (scale: s_in, zp: z_in)
    ↓ DequantizeLinear
FP32
    ↓ Conv2D(16, 3x3) + BN + ReLU
FP32 activations
    ↓ QuantizeLinear(scale: s_mid, zp: z_mid)
int8
    ↓ DequantizeLinear
FP32
    ↓ Conv2D(16, 3x3) + BN
FP32 (pre-add)
    ↓ QuantizeLinear(scale: s_main, zp: z_main)
int8 (ready for add)
```

**Stage 4: Residual Block (Skip Path)**

```
Input to residual block: int8 (scale: s_in, zp: z_in)
    ↓ DequantizeLinear
FP32
    ↓ [Identity or Conv1x1 if dimension mismatch]
FP32
    ↓ QuantizeLinear(scale: s_skip, zp: z_skip)
int8 (ready for add)
```

**Stage 5: Residual Addition (CRITICAL POINT)**

```
Main path:  int8 (scale: s_main, zp: z_main)
Skip path:  int8 (scale: s_skip, zp: z_skip)

Problem: Scales may differ!

Solution: Dequantize both, add in FP32, requantize

Main path:
    ↓ DequantizeLinear
FP32 (scale: s_main)

Skip path:
    ↓ DequantizeLinear
FP32 (scale: s_skip)

    ↓ Add (element-wise, FP32)
FP32 sum
    ↓ ReLU (FP32)
FP32 activations
    ↓ QuantizeLinear(scale: s_out, zp: z_out)
int8 (output of residual block)
```

**Hardware perspective for Add:**
```
If s_main ≈ s_skip (scales matched):
  1. Add int8 tensors directly: result_i8 = x_main + x_skip
  2. Adjust for zero-point differences if needed
  3. Requantize to output scale

If s_main ≠ s_skip (scales differ):
  1. Dequantize both: x_main_fp32 = (x_main - z_main) * s_main
                      x_skip_fp32 = (x_skip - z_skip) * s_skip
  2. Add in FP32: result_fp32 = x_main_fp32 + x_skip_fp32
  3. Quantize result: result_i8 = result_fp32 / s_out + z_out
```

**Stage 6: Output Classification Head**

```
Final residual block output: int8
    ↓ DequantizeLinear
FP32
    ↓ GlobalAveragePooling (FP32)
FP32 (1, 64)
    ↓ Dense(10) / Linear (FP32 or quantized)
FP32 (1, 10) logits
    ↓ Softmax (FP32, typically not quantized)
FP32 (1, 10) probabilities
```

### Data Type Transitions

```
Stage                     Data Type    Scale/ZP              Notes
─────────────────────────────────────────────────────────────────────
Input (raw pixels)        FP32 [0,255] N/A                   Original
  ↓ QuantizeLinear
Input (quantized)         uint8        s=1.0, zp=0           Boundary
  ↓ DequantizeLinear
Conv input                FP32         N/A                   Compute
  ↓ Conv + Quantize
Conv output (quantized)   int8         s=s1, zp=z1           Activation
  ↓ DequantizeLinear
ReLU input                FP32         N/A                   Compute
  ↓ ReLU + Quantize
ReLU output (quantized)   int8         s=s2, zp=z2           Activation
  ... (repeated through residual blocks)
Final quantized           int8         s=s_final, zp=z_final Last activation
  ↓ DequantizeLinear
Classification head       FP32         N/A                   Output boundary
```

## Residual Connection Quantization

### The Scale Mismatch Problem

**Problem:** Residual connections add two tensors that may have different scales.

```
Main path:    y_main = (x_main - z_main) * s_main
Skip path:    y_skip = (x_skip - z_skip) * s_skip

Cannot add directly if s_main ≠ s_skip!
```

**Why scales differ:**
- Main path goes through Conv → BN → ReLU → Conv → BN
- Skip path is identity or Conv1x1 → BN
- Different operations produce different activation ranges
- Calibration learns different scales for each path

### Solution 1: Requantization Before Add (ONNX Runtime QDQ)

**Approach:** Dequantize both paths, add in FP32, requantize output.

```
┌─────────────────────┐        ┌─────────────────────┐
│ Main Path (int8)    │        │ Skip Path (int8)    │
│ scale: s_main       │        │ scale: s_skip       │
│ zp: z_main          │        │ zp: z_skip          │
└──────────┬──────────┘        └──────────┬──────────┘
           │                              │
           v                              v
    DequantizeLinear               DequantizeLinear
           │                              │
           v                              v
       FP32 (s_main)                  FP32 (s_skip)
           │                              │
           └──────────────┬───────────────┘
                          v
                     Add (FP32)
                          │
                          v
                     FP32 result
                          │
                          v
                  QuantizeLinear
                  (scale: s_out, zp: z_out)
                          │
                          v
                    int8 output
```

**Hardware implementation:**
```c++
// Dequantize main path
float main_fp[SIZE];
for (int i = 0; i < SIZE; i++) {
    main_fp[i] = (main_int8[i] - z_main) * s_main;
}

// Dequantize skip path
float skip_fp[SIZE];
for (int i = 0; i < SIZE; i++) {
    skip_fp[i] = (skip_int8[i] - z_skip) * s_skip;
}

// Add in FP32
float result_fp[SIZE];
for (int i = 0; i < SIZE; i++) {
    result_fp[i] = main_fp[i] + skip_fp[i];
}

// Quantize output
int8_t result_int8[SIZE];
for (int i = 0; i < SIZE; i++) {
    result_int8[i] = saturate_i8(round(result_fp[i] / s_out + z_out));
}
```

**Cost:** 2 dequantizations + 1 quantization = 3× scale/offset operations

### Solution 2: Scale Matching (TensorRT Optimization)

**Approach:** Requantize one path to match the other's scale, then add in integer.

```
Main path: int8 (scale: s_main)
Skip path: int8 (scale: s_skip)

If s_main < s_skip (finer quantization on main path):
  1. Requantize skip to match main:
     skip_requant = round((skip - z_skip) * s_skip / s_main) + z_main

  2. Add in int8:
     result = main + skip_requant - z_main  (adjust for double zero-point)

  3. Output already at scale s_main

If s_main > s_skip:
  (requantize main to match skip)
```

**Hardware implementation:**
```c++
// Requantize skip path to main path scale
int8_t skip_requant[SIZE];
for (int i = 0; i < SIZE; i++) {
    // Scale conversion: s_skip → s_main
    int32_t temp = (skip_int8[i] - z_skip) * (s_skip / s_main);
    skip_requant[i] = saturate_i8(round(temp + z_main));
}

// Add in integer
int8_t result_int8[SIZE];
for (int i = 0; i < SIZE; i++) {
    int32_t sum = main_int8[i] + skip_requant[i] - z_main;
    result_int8[i] = saturate_i8(sum);
}
```

**Cost:** 1 requantization + integer add (cheaper than full dequant/quant cycle)

### Solution 3: PyTorch FloatFunctional

**Approach:** Use special quantized add operation that handles scale/zp internally.

```python
# PyTorch quantized ResNet implementation
self.skip_add = torch.nn.quantized.FloatFunctional()

# In forward pass
out = self.skip_add.add(main_path, skip_path)
```

**What FloatFunctional.add() does:**
1. Checks scales of both inputs
2. If scales match: direct integer add
3. If scales differ: requantizes one path or dequantizes both
4. Returns quantized tensor with appropriate scale

**Source:** [PyTorch vision quantized ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py)

### TensorRT Optimization Strategy

**Goal:** Maximize integer operations, minimize precision conversions.

**Strategy:** Insert Q/DQ nodes strategically to enable INT8 fusion.

```
Original (inefficient):
Conv → Q → DQ → Add (FP32) → Q → DQ → Conv

Optimized (TensorRT):
Conv → Q ──────┐
               │ (both int8, scales matched)
Skip ──────────┘
               │
               v
          Add (INT8)
               │
               v
          DQ → Conv
```

**How TensorRT achieves this:**
1. Propagate Q nodes backward through the graph
2. Propagate DQ nodes forward through the graph
3. Requantize residual inputs to match scales
4. Fuse Q/DQ with weighted operations (Conv, MatMul)

**Result:** Residual Add operates in INT8, eliminating FP32 conversion overhead.

**Source:** [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)

## QLinear Operation Internals

### QLinearConv: Integer Convolution

**Inputs:**
```
1. x: uint8/int8 input tensor (B, H, W, C_in)
2. x_scale: float (scalar or per-channel)
3. x_zero_point: uint8/int8 (scalar or per-channel)
4. w: int8 weight tensor (C_out, K_h, K_w, C_in)
5. w_scale: float (scalar or per-channel)
6. w_zero_point: int8 (scalar or per-channel)
7. y_scale: float (output scale)
8. y_zero_point: uint8/int8 (output zero-point)
9. b: int32 bias (optional, quantized with scale = x_scale * w_scale)
```

**Output:**
```
y: uint8/int8 output tensor (B, H_out, W_out, C_out)
```

**Integer Arithmetic (per output position):**

```
Step 1: Integer convolution (accumulate in int32)
  acc = 0
  for each kernel position (kh, kw, c_in):
    x_val = x[h+kh, w+kw, c_in] - x_zero_point
    w_val = w[c_out, kh, kw, c_in] - w_zero_point
    acc += x_val * w_val

  if bias exists:
    acc += b[c_out]

Step 2: Requantization (scale conversion)
  scale_factor = (x_scale * w_scale) / y_scale
  acc_scaled = acc * scale_factor

Step 3: Add output zero-point and saturate
  y[h_out, w_out, c_out] = saturate(round(acc_scaled) + y_zero_point)
```

**Hardware-friendly pseudocode:**

```c++
int32_t compute_qconv_pixel(
    const uint8_t* x, const int8_t* w, const int32_t* bias,
    int h_out, int w_out, int c_out,
    float x_scale, uint8_t x_zp,
    float w_scale, int8_t w_zp,
    float y_scale, uint8_t y_zp
) {
    // Accumulate in int32 (wide enough for 3x3x64 kernel)
    int32_t acc = 0;

    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            for (int c_in = 0; c_in < channels_in; c_in++) {
                int h_in = h_out * stride + kh;
                int w_in = w_out * stride + kw;

                // Subtract zero-points
                int32_t x_val = (int32_t)x[h_in, w_in, c_in] - x_zp;
                int32_t w_val = (int32_t)w[c_out, kh, kw, c_in] - w_zp;

                // Accumulate
                acc += x_val * w_val;
            }
        }
    }

    // Add bias (already quantized with scale = x_scale * w_scale)
    if (bias) {
        acc += bias[c_out];
    }

    // Requantization: scale conversion
    float scale_factor = (x_scale * w_scale) / y_scale;
    float acc_scaled = (float)acc * scale_factor;

    // Round, add output zero-point, saturate to uint8 range
    int32_t result = (int32_t)round(acc_scaled) + y_zp;
    return saturate_u8(result);  // Clip to [0, 255]
}
```

**Per-Channel Quantization:**

If using per-channel weight quantization (better accuracy):

```c++
// Each output channel has its own w_scale and w_zp
float w_scale[c_out];
int8_t w_zp[c_out];

// In computation:
for (int oc = 0; oc < c_out; oc++) {
    float scale_factor = (x_scale * w_scale[oc]) / y_scale;
    // ... rest of computation using w_scale[oc] and w_zp[oc]
}
```

### QLinearMatMul: Integer Matrix Multiplication

**Inputs:**
```
1. a: uint8/int8 matrix (M, K)
2. a_scale: float
3. a_zero_point: uint8/int8
4. b: uint8/int8 matrix (K, N)
5. b_scale: float
6. b_zero_point: uint8/int8
7. y_scale: float
8. y_zero_point: uint8/int8
```

**Output:**
```
y: uint8/int8 matrix (M, N)
```

**Integer Arithmetic:**

```c++
// For each output element y[m, n]
int32_t acc = 0;
for (int k = 0; k < K; k++) {
    int32_t a_val = (int32_t)a[m, k] - a_zero_point;
    int32_t b_val = (int32_t)b[k, n] - b_zero_point;
    acc += a_val * b_val;
}

// Requantize
float scale_factor = (a_scale * b_scale) / y_scale;
float acc_scaled = (float)acc * scale_factor;
int32_t result = (int32_t)round(acc_scaled) + y_zero_point;
y[m, n] = saturate(result);
```

**Optimizations for hardware:**

1. **SIMD vectorization:** Process 8 or 16 int8 multiplications in parallel
2. **INT8 → INT16 → INT32 accumulation:** Avoid overflow
3. **Fused multiply-add (FMA):** acc += a * b in one instruction
4. **Block-wise computation:** Tile M, N, K dimensions for cache efficiency

### QuantizeLinear: FP32 → INT8

**Formula:**
```
y_quantized = saturate(round(x_float / y_scale) + y_zero_point)
```

**Hardware implementation:**

```c++
int8_t quantize_linear(
    float x,
    float scale,
    int8_t zero_point
) {
    // Divide by scale
    float scaled = x / scale;

    // Round to nearest even (banker's rounding)
    int32_t rounded = (int32_t)roundf(scaled);

    // Add zero-point
    int32_t with_zp = rounded + zero_point;

    // Saturate to int8 range [-128, 127]
    if (with_zp < -128) return -128;
    if (with_zp > 127) return 127;
    return (int8_t)with_zp;
}

// Vectorized version (SIMD):
void quantize_linear_simd(
    const float* x,
    int8_t* y,
    size_t size,
    float scale,
    int8_t zero_point
) {
    float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < size; i += 8) {
        // Load 8 floats
        __m256 x_vec = _mm256_loadu_ps(&x[i]);

        // Multiply by 1/scale
        __m256 scaled = _mm256_mul_ps(x_vec, _mm256_set1_ps(inv_scale));

        // Round
        __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT);

        // Convert to int32
        __m256i int32_vec = _mm256_cvtps_epi32(rounded);

        // Add zero-point
        __m256i with_zp = _mm256_add_epi32(int32_vec, _mm256_set1_epi32(zero_point));

        // Pack to int8 with saturation
        __m128i int8_vec = _mm256_cvtsepi32_epi8(with_zp);

        // Store
        _mm_storeu_si64(&y[i], int8_vec);
    }
}
```

### DequantizeLinear: INT8 → FP32

**Formula:**
```
y_float = (x_quantized - x_zero_point) * x_scale
```

**Hardware implementation:**

```c++
float dequantize_linear(
    int8_t x,
    float scale,
    int8_t zero_point
) {
    // Subtract zero-point
    int32_t x_shifted = (int32_t)x - zero_point;

    // Multiply by scale
    float result = (float)x_shifted * scale;

    return result;
}

// Vectorized version (SIMD):
void dequantize_linear_simd(
    const int8_t* x,
    float* y,
    size_t size,
    float scale,
    int8_t zero_point
) {
    for (size_t i = 0; i < size; i += 8) {
        // Load 8 int8 values
        __m128i x_vec = _mm_loadu_si64(&x[i]);

        // Sign-extend to int32
        __m256i int32_vec = _mm256_cvtepi8_epi32(x_vec);

        // Subtract zero-point
        __m256i shifted = _mm256_sub_epi32(int32_vec, _mm256_set1_epi32(zero_point));

        // Convert to float
        __m256 float_vec = _mm256_cvtepi32_ps(shifted);

        // Multiply by scale
        __m256 result = _mm256_mul_ps(float_vec, _mm256_set1_ps(scale));

        // Store
        _mm256_storeu_ps(&y[i], result);
    }
}
```

## Where Scale and Zero-Point Parameters Live

### In ONNX Graph (QDQ Format)

**QuantizeLinear node:**
```
Node: QuantizeLinear
  Inputs:
    - x: tensor (FP32)
    - y_scale: tensor (float32 scalar or 1D)
    - y_zero_point: tensor (int8/uint8 scalar or 1D)
  Outputs:
    - y: tensor (int8/uint8)
```

**Example ONNX graph snippet:**
```protobuf
node {
  input: "conv1_output"
  input: "QuantizeLinear_scale_1"    # Constant tensor
  input: "QuantizeLinear_zp_1"       # Constant tensor
  output: "conv1_quantized"
  op_type: "QuantizeLinear"
}

initializer {
  name: "QuantizeLinear_scale_1"
  dims: []  # Scalar
  data_type: FLOAT
  float_data: 0.0234567  # Learned during calibration
}

initializer {
  name: "QuantizeLinear_zp_1"
  dims: []  # Scalar
  data_type: INT8
  int32_data: 0  # Symmetric quantization
}
```

**Storage location:** Graph initializers (constant tensors embedded in .onnx file)

### In ONNX Graph (QOperator Format)

**QLinearConv node:**
```
Node: QLinearConv
  Inputs:
    - x: tensor (uint8/int8)
    - x_scale: tensor (float32 scalar)
    - x_zero_point: tensor (uint8/int8 scalar)
    - w: tensor (int8 weights)
    - w_scale: tensor (float32 scalar or 1D per-channel)
    - w_zero_point: tensor (int8 scalar or 1D per-channel)
    - y_scale: tensor (float32 scalar)
    - y_zero_point: tensor (uint8/int8 scalar)
    - [optional] bias: tensor (int32)
  Outputs:
    - y: tensor (uint8/int8)
```

**Example:**
```protobuf
node {
  input: "input_quantized"
  input: "x_scale"
  input: "x_zero_point"
  input: "conv1_weight_quantized"
  input: "w_scale"
  input: "w_zero_point"
  input: "y_scale"
  input: "y_zero_point"
  input: "conv1_bias"
  output: "conv1_output_quantized"
  op_type: "QLinearConv"
  attribute {
    name: "kernel_shape"
    ints: [3, 3]
  }
  attribute {
    name: "strides"
    ints: [1, 1]
  }
}
```

**Storage location:** Mix of initializers (weights, scales) and runtime tensors (activations)

### In PyTorch Quantized Model

**Quantized Conv2d module:**
```python
# PyTorch quantized module
class QuantizedConv2d(torch.nn.Module):
    def __init__(self):
        self.weight = torch.quantized_tensor(...)  # int8 with built-in scale/zp
        self.bias = torch.tensor(...)              # int32 bias
        self.scale = 0.0234567                     # Output scale
        self.zero_point = 0                        # Output zero-point
```

**Accessing parameters:**
```python
import torch

# Load quantized model
model = torch.load('resnet8_int8.pt')

# Access quantized weight
conv1_weight = model.conv1.weight()
print(f"Weight dtype: {conv1_weight.dtype}")        # torch.qint8
print(f"Weight scale: {conv1_weight.q_scale()}")   # 0.0156 (example)
print(f"Weight zp: {conv1_weight.q_zero_point()}")  # 0 (symmetric)

# Access quantized activation scale/zp
print(f"Output scale: {model.conv1.scale}")
print(f"Output zp: {model.conv1.zero_point}")
```

**Storage location:** Model state_dict with embedded quantization parameters

### Hardware Accelerator Memory Layout

**What a hardware accelerator needs:**

```c
// Per-layer quantization parameters (stored in SRAM/DRAM)
struct QuantParams {
    float input_scale;
    int8_t input_zp;
    float weight_scale[MAX_CHANNELS];   // Per-channel
    int8_t weight_zp[MAX_CHANNELS];     // Per-channel
    float output_scale;
    int8_t output_zp;
};

// Convolution layer descriptor
struct ConvLayer {
    // Weights (int8, stored in DRAM)
    int8_t* weights;              // (C_out, K_h, K_w, C_in)
    int32_t* bias;                // (C_out) quantized

    // Quantization params
    QuantParams quant;

    // Layer config
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int in_channels, out_channels;
};

// Execution (hardware accelerator pseudocode)
void execute_qconv(
    const ConvLayer* layer,
    const int8_t* input,     // Quantized input
    int8_t* output           // Quantized output
) {
    // For each output position
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            for (int oc = 0; oc < layer->out_channels; oc++) {
                // Compute using layer->quant parameters
                output[h, w, oc] = compute_qconv_pixel(
                    input, layer->weights, layer->bias,
                    h, w, oc,
                    layer->quant.input_scale, layer->quant.input_zp,
                    layer->quant.weight_scale[oc], layer->quant.weight_zp[oc],
                    layer->quant.output_scale, layer->quant.output_zp
                );
            }
        }
    }
}
```

## PyTorch Quantized Operations

### PyTorch Quantization Overview

**Data types:**
- `torch.qint8`: Signed 8-bit integer (-128 to 127)
- `torch.quint8`: Unsigned 8-bit integer (0 to 255)
- `torch.qint32`: 32-bit integer (for bias)

**Quantized tensor structure:**
```python
# Create quantized tensor
q_tensor = torch.quantize_per_tensor(
    float_tensor,
    scale=0.1,
    zero_point=0,
    dtype=torch.qint8
)

# Access quantization parameters
scale = q_tensor.q_scale()         # 0.1
zero_point = q_tensor.q_zero_point()  # 0

# Get underlying integer representation
int_repr = q_tensor.int_repr()     # int8 values
```

### PyTorch Quantized Conv2d

**Module definition:**
```python
# FP32 conv
conv_fp32 = torch.nn.Conv2d(16, 32, kernel_size=3)

# Quantized conv (after static quantization)
conv_q = torch.ao.nn.quantized.Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3
)

# Quantized conv stores:
conv_q.weight()  # torch.qint8 tensor with embedded scale/zp
conv_q.scale     # Output scale
conv_q.zero_point  # Output zero-point
```

**Computation:**
```python
# Forward pass
x_quantized = torch.quantize_per_tensor(x_fp32, scale=0.1, zero_point=0, dtype=torch.quint8)
y_quantized = conv_q(x_quantized)  # Output is also quantized

# Internally performs:
# 1. Integer convolution: int8 weights × quint8 input → int32 accumulator
# 2. Requantization: int32 → quint8 with output scale/zp
```

### PyTorch FloatFunctional (for residual connections)

**Purpose:** Handle quantized tensor addition with automatic scale matching.

```python
# In quantized ResNet block
class QuantizedResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.ao.nn.quantized.Conv2d(...)
        self.conv2 = torch.ao.nn.quantized.Conv2d(...)
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = torch.nn.functional.relu(out, inplace=True)
        out = self.conv2(out)

        # Quantized addition (handles scale matching)
        out = self.add.add(out, identity)
        out = torch.nn.functional.relu(out, inplace=True)

        return out
```

**What FloatFunctional.add() does:**
```python
# Simplified implementation
class FloatFunctional:
    def add(self, x_quantized, y_quantized):
        # Get scales and zero-points
        x_scale = x_quantized.q_scale()
        x_zp = x_quantized.q_zero_point()
        y_scale = y_quantized.q_scale()
        y_zp = y_quantized.q_zero_point()

        if x_scale == y_scale and x_zp == y_zp:
            # Direct integer addition (fast path)
            result_int = x_quantized.int_repr() + y_quantized.int_repr()
            return torch.quantize_per_tensor(
                result_int, scale=x_scale, zero_point=x_zp, dtype=torch.qint8
            )
        else:
            # Requantization (slow path)
            x_dequant = x_quantized.dequantize()
            y_dequant = y_quantized.dequantize()
            result_fp32 = x_dequant + y_dequant

            # Choose output scale (typically max of inputs)
            output_scale = max(x_scale, y_scale)
            output_zp = 0

            return torch.quantize_per_tensor(
                result_fp32, scale=output_scale, zero_point=output_zp, dtype=torch.qint8
            )
```

### PyTorch vs ONNX Quantization Equivalence

| PyTorch | ONNX (QDQ) | ONNX (QOperator) |
|---------|------------|------------------|
| `torch.quantize_per_tensor()` | `QuantizeLinear` | `QuantizeLinear` |
| `tensor.dequantize()` | `DequantizeLinear` | `DequantizeLinear` |
| `torch.ao.nn.quantized.Conv2d` | Conv + Q/DQ wrap | `QLinearConv` |
| `torch.ao.nn.quantized.Linear` | MatMul + Q/DQ wrap | `QLinearMatMul` |
| `FloatFunctional.add()` | Add + Q/DQ | Add (with scale handling) |

**Conversion flow:**
```
PyTorch quantized model
    ↓ torch.onnx.export()
ONNX model (QDQ format)
    ↓ onnx.optimizer
Optimized ONNX (operator fusion)
    ↓ onnxruntime.quantization.qdq_to_qop (optional)
ONNX model (QOperator format)
```

## Hardware Accelerator Implementation Guide

### What a Hardware Accelerator Must Implement

**Core operations:**

1. **QuantizeLinear** (FP32 → INT8)
   - Division by scale (FP32 or fixed-point approximation)
   - Rounding (to nearest even)
   - Addition of zero-point
   - Saturation to int8 range

2. **DequantizeLinear** (INT8 → FP32)
   - Subtraction of zero-point
   - Multiplication by scale (FP32 or fixed-point)

3. **QLinearConv** (INT8 convolution)
   - Integer MAC (multiply-accumulate) in int32
   - Requantization (scale conversion)
   - Bias addition
   - Saturation

4. **Add** (INT8 addition with scale handling)
   - Either: Direct int8 add if scales match
   - Or: Dequant → FP32 add → Quant if scales differ

5. **ReLU** (INT8 activation)
   - max(x, zero_point)

**Optional optimizations:**

6. **Fused QConv-ReLU**: Combine convolution + ReLU in one operation
7. **Per-channel quantization**: Support different scales per output channel
8. **Dynamic requantization**: Adjust scales on-the-fly for scale matching

### Datapath Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Hardware Accelerator Datapath                 │
└──────────────────────────────────────────────────────────────────┘

Input Memory (DRAM)
    ↓
┌─────────────────────┐
│ Input Buffer (SRAM) │  ← FP32 input (optional DMA)
└─────────────────────┘
    ↓
┌─────────────────────┐
│ QuantizeLinear Unit │  ← FP32 → INT8 conversion
│   - FP divider      │
│   - Rounder         │
│   - Saturator       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ INT8 Feature Buffer │  ← Quantized activations (SRAM)
└─────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ MAC Array (Systolic or SIMD)            │
│   - INT8 × INT8 → INT32 multiply        │
│   - INT32 accumulate                    │
│   - Weight buffer (SRAM)                │
│   - Per-channel scale/zp storage        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────┐
│ Requantization Unit │  ← INT32 → INT8 conversion
│   - FP multiplier   │     (scale conversion)
│   - Rounder         │
│   - Saturator       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Activation Unit     │  ← ReLU, Add (for residual)
│   - ReLU            │
│   - INT8 Adder      │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Output Buffer (SRAM)│  ← INT8 activations
└─────────────────────┘
    ↓
┌─────────────────────┐
│ DequantizeLinear    │  ← INT8 → FP32 (for final output)
│   - FP multiplier   │
└─────────────────────┘
    ↓
Output Memory (DRAM)
```

### Computation Pipeline

**Example: ResNet8 residual block execution**

```
Cycle 1-N: Load input activations (INT8) from DRAM → SRAM
Cycle N+1: Start Conv1 MAC operations
  - Load weights for Conv1 (INT8)
  - Compute: INT8 × INT8 → INT32 accumulator
  - Requantize: INT32 → INT8 with scale conversion
  - Store intermediate activations

Cycle M: ReLU activation (INT8)
  - max(x, zero_point)

Cycle M+1: Start Conv2 MAC operations
  - Load weights for Conv2
  - Compute INT8 × INT8 → INT32
  - Requantize → INT8

Cycle P: Residual Add
  - Load skip path activations (INT8)
  - Check scales: match? → INT8 add : FP32 add
  - Store result

Cycle P+1: ReLU activation
  - Output ready for next block
```

### Memory Requirements

**For ResNet8 (one residual block):**

```
Input activations: (1, 16, 16, 32) × 1 byte = 8 KB (INT8)
Weights (Conv 3×3): (32, 3, 3, 32) × 1 byte = 9 KB (INT8)
Bias: (32) × 4 bytes = 128 bytes (INT32)
Output activations: (1, 16, 16, 32) × 1 byte = 8 KB (INT8)

Scale/zero-point per layer:
  Input: 2 × 4 bytes = 8 bytes (scale + zp)
  Weights: 32 channels × 2 × 4 bytes = 256 bytes (per-channel)
  Output: 2 × 4 bytes = 8 bytes

Total per block: ~17 KB (weights) + ~8 KB (activations) = ~25 KB
```

**Full ResNet8:**
- Total weights: ~315 KB (FP32) → ~79 KB (INT8, 4× reduction)
- Peak activation memory: ~64 KB (largest feature maps)
- Scale/zero-point storage: ~2 KB (all layers)

### Quantization Parameter Storage

**Accelerator memory layout:**

```c
// Compact parameter storage for hardware
typedef struct {
    // Weight memory (INT8 tensor)
    int8_t weights[MAX_WEIGHTS];

    // Per-channel scales (FP32 or fixed-point)
    float weight_scales[MAX_CHANNELS];
    uint8_t weight_zero_points[MAX_CHANNELS];

    // Bias (INT32, quantized with scale = input_scale × weight_scale)
    int32_t bias[MAX_CHANNELS];

    // Activation scales (per-layer)
    float input_scale;
    uint8_t input_zero_point;
    float output_scale;
    uint8_t output_zero_point;

} QuantizedLayer;
```

**Loading parameters:**
```c
// At runtime, load from ONNX model or PyTorch checkpoint
void load_layer_params(QuantizedLayer* layer, const char* model_path) {
    // Parse ONNX graph or PyTorch state_dict
    // Extract weight tensors (INT8)
    // Extract scale/zero-point initializers
    // Store in accelerator SRAM/DRAM
}
```

## Graph Visualization Examples

### QDQ Format Detailed View

```
ResNet8 Quantized (QDQ format) - Single Residual Block

Input: int8 (scale: 0.05, zp: 0)
│
├─────────────────────────────┐
│                             │
│ MAIN PATH                   │ SKIP PATH
│                             │
v                             v
DequantizeLinear              DequantizeLinear
│                             │
v                             v
FP32                          FP32 (identity)
│                             │
v                             │
Conv2D(32, 3×3)               │
│  Weights: int8              │
│  W_scale: [s_w1, ..., s_w32]│
│  W_zp: [0, ..., 0]          │
│                             │
v                             │
FP32 activations              │
│                             │
v                             │
QuantizeLinear                │
│  scale: 0.03                │
│  zp: 0                      │
│                             │
v                             │
int8                          │
│                             │
v                             │
DequantizeLinear              │
│                             │
v                             │
FP32                          │
│                             │
v                             │
BatchNorm (fused)             │
│                             │
v                             │
ReLU                          │
│                             │
v                             │
QuantizeLinear                │
│  scale: 0.04                │
│  zp: 0                      │
│                             │
v                             │
int8                          │
│                             │
v                             │
DequantizeLinear              │
│                             │
v                             │
FP32                          │
│                             │
v                             │
Conv2D(32, 3×3)               │
│  Weights: int8              │
│                             │
v                             │
FP32                          │
│                             │
v                             │
BatchNorm (fused)             │
│                             │
v                             │
QuantizeLinear                QuantizeLinear
│  scale: 0.05                │  scale: 0.05
│  zp: 0                      │  zp: 0
│                             │
v                             v
int8                          int8
│                             │
v                             v
DequantizeLinear              DequantizeLinear
│                             │
└──────────┬──────────────────┘
           │
           v
       Add (FP32)
           │
           v
       FP32 result
           │
           v
       ReLU
           │
           v
       QuantizeLinear
       │  scale: 0.06
       │  zp: 0
       v
       int8 output
```

### QOperator Format Detailed View

```
ResNet8 Quantized (QOperator format) - Single Residual Block

Input: int8 (scale: 0.05, zp: 0)
│
├─────────────────────────────┐
│                             │
│ MAIN PATH                   │ SKIP PATH
│                             │
v                             v
┌──────────────────────────┐  (identity, stays int8)
│ QLinearConv              │  int8 (scale: 0.05, zp: 0)
│   x: int8                │  │
│   x_scale: 0.05          │  │
│   x_zp: 0                │  │
│   w: int8 weights        │  │
│   w_scale: [...]         │  │
│   w_zp: [...]            │  │
│   y_scale: 0.03          │  │
│   y_zp: 0                │  │
│   bias: int32            │  │
└──────────────────────────┘  │
│                             │
v                             │
int8 (scale: 0.03, zp: 0)     │
│                             │
v                             │
[BatchNorm fused into conv]   │
│                             │
v                             │
[ReLU: max(x, 0)]             │
│                             │
v                             │
┌──────────────────────────┐  │
│ QLinearConv (2nd)        │  │
│   x: int8                │  │
│   x_scale: 0.03          │  │
│   x_zp: 0                │  │
│   w: int8 weights        │  │
│   w_scale: [...]         │  │
│   w_zp: [...]            │  │
│   y_scale: 0.05          │  │
│   y_zp: 0                │  │
└──────────────────────────┘  │
│                             │
v                             v
int8 (scale: 0.05, zp: 0)     int8 (scale: 0.05, zp: 0)
│                             │
└──────────┬──────────────────┘
           │
           v (scales match!)
       ┌────────────┐
       │ INT8 Add   │  ← Direct integer addition
       └────────────┘
           │
           v
       int8 (scale: 0.05, zp: 0)
           │
           v
       [ReLU: max(x, 0)]
           │
           v
       int8 output
```

## Confidence Assessment

| Aspect | Confidence | Rationale |
|--------|-----------|-----------|
| **ONNX quantization formulas** | HIGH | Official ONNX operator spec documentation |
| **QDQ vs QOperator structure** | HIGH | ONNX Runtime documentation and examples |
| **Residual connection handling** | HIGH | TensorRT and PyTorch documentation with examples |
| **QLinearConv internals** | HIGH | ONNX operator spec with formula details |
| **PyTorch quantization equivalence** | MEDIUM-HIGH | PyTorch docs, but specifics vary by version |
| **Hardware accelerator requirements** | MEDIUM | Industry standard approach, but implementation-specific |
| **Scale matching optimization** | MEDIUM-HIGH | TensorRT documentation, but vendor-specific details |

## Sources

### High Confidence (Official Documentation)

- [QuantizeLinear - ONNX](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) - Quantization operator specification
- [DequantizeLinear - ONNX](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) - Dequantization operator specification
- [QLinearConv - ONNX](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) - Integer convolution specification
- [QLinearMatMul - ONNX](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) - Integer matrix multiplication
- [Quantize ONNX models - ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - QDQ vs QOperator formats
- [Working with Quantized Types - NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html) - Residual connection optimization
- [PyTorch Quantized ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py) - FloatFunctional.add() implementation
- [PyTorch Introducing Quantized Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor) - PyTorch quantization fundamentals
- [PyTorch Quantization API](https://docs.pytorch.org/docs/stable/quantization.html) - Official quantization documentation

### Medium Confidence (Community and Research)

- [NVIDIA TensorRT Quantization Toolkit](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/tensorflow-quantization-toolkit/docs/index.html) - QDQ placement strategies
- [Quantization Support In ONNX](https://github.com/onnx/onnx/wiki/Quantization-Support-In-ONNX) - Historical context and design decisions

### Low Confidence (Inferred or Implementation-Specific)

- Hardware accelerator specific implementations vary by vendor
- Optimal scale matching strategies depend on hardware capabilities
- Per-channel quantization support varies by runtime

## Notes for Documentation Development

This architecture research provides the foundation for creating hardware accelerator implementation documentation. Key areas to document in v1.3:

1. **QLinearConv calculation step-by-step** - With concrete ResNet8 examples
2. **QuantizeLinear/DequantizeLinear boundary operations** - Input/output conversion
3. **Residual connection scale matching** - Three solution approaches
4. **PyTorch quantized operation equivalents** - Map to ONNX concepts
5. **ONNX graph visualization** - Annotated Netron-style diagrams

**Critical for hardware implementers:**
- Integer arithmetic formulas (ready to implement)
- Scale/zero-point parameter locations in graph
- Memory layout recommendations
- Residual connection handling strategies

**Phase structure implication:** Documentation can be created layer-by-layer (conv → residual block → full network) to build understanding incrementally.
