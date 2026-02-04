# Feature Landscape: Quantized Operations for Hardware Accelerators

**Domain:** Quantized Neural Network Operations
**Researched:** 2026-02-02
**Project:** ResNet8 Hardware Accelerator Documentation
**Milestone:** v1.3 Quantized Operations Documentation
**Confidence:** HIGH

## Executive Summary

This research documents the exact mathematical operations required for implementing quantized neural network inference in hardware accelerators. The focus is on ONNX quantized operators (QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear) and their PyTorch equivalents, with emphasis on the integer arithmetic that hardware must implement.

Hardware accelerators need to implement 8-bit integer arithmetic with 32-bit accumulators, scale/zero-point parameters, and saturation logic. The operations fall into three categories: quantization/dequantization (FP32 ↔ INT8), quantized convolution, and quantized matrix multiplication.

---

## Table Stakes: Core Quantized Operations

These are the fundamental operations any quantized neural network hardware accelerator must implement.

### 1. QuantizeLinear (FP32 → INT8/UINT8)

**Purpose:** Convert floating-point tensors to quantized integer representation

**Mathematical Formula:**
```
y = saturate(round(x / y_scale) + y_zero_point)
```

**Detailed Specification:**
- **Rounding mode:** Round to nearest even (banker's rounding, IEEE 754)
- **Saturation ranges:**
  - INT8: [-128, 127]
  - UINT8: [0, 255]
- **Input:** x (FP32, FP16, BF16)
- **Output:** y (INT8 or UINT8)

**Hardware Requirements:**
- Division operation (x / y_scale)
- Round-to-nearest-even logic
- Integer addition (+ y_zero_point)
- Saturation/clipping to target range

**Quantization Granularities:**

| Granularity | Scale Shape | Zero Point Shape | Use Case |
|-------------|-------------|------------------|----------|
| Per-tensor | Scalar | Scalar | Activations, simple quantization |
| Per-channel | 1-D tensor (length = channels) | 1-D tensor | Convolution weights |
| Blocked | Matches input except 1 dim | Matches scale | Advanced quantization |

**Parameter Calculation:**

For per-tensor quantization:
```
scale = (rmax - rmin) / (qmax - qmin)

where:
  rmax, rmin = max and min of floating-point tensor
  qmax, qmin = max and min of quantized type
    - INT8:  qmin = -128, qmax = 127
    - UINT8: qmin = 0,    qmax = 255

zero_point = round(qmin - rmin / scale)
zero_point = clip(zero_point, qmin, qmax)
```

**ONNX Specification:** Version 25+
**Confidence:** HIGH (verified with official ONNX documentation)

---

### 2. DequantizeLinear (INT8/UINT8 → FP32)

**Purpose:** Convert quantized integer tensors back to floating-point representation

**Mathematical Formula:**
```
y = (x - x_zero_point) * x_scale
```

**Detailed Specification:**
- **Input:** x (INT8, UINT8, INT16, UINT16, INT32)
- **Output:** y (FP32, FP16, BF16)
- **Shape preservation:** Output has same shape as input

**Hardware Requirements:**
- Integer subtraction (x - x_zero_point)
- Floating-point multiplication (* x_scale)
- Type conversion (INT → FP)

**Quantization Granularities:**
Same as QuantizeLinear (per-tensor, per-channel, blocked)

**Key Constraint:**
- x_zero_point and x must have the same type (both INT8 or both UINT8)

**ONNX Specification:** Version 25+
**Confidence:** HIGH (verified with official ONNX documentation)

---

### 3. QLinearConv (Quantized Convolution)

**Purpose:** Perform convolution operation entirely in quantized integer space

**High-Level Formula:**
```
QLinearConv = ConvInteger + QuantizeLinear
```

**Inputs (8-9 total):**
1. `x` - quantized input (INT8/UINT8)
2. `x_scale` - input scale factor
3. `x_zero_point` - input zero point
4. `w` - quantized weights (INT8/UINT8)
5. `w_scale` - weight scale factor
6. `w_zero_point` - weight zero point
7. `y_scale` - output scale factor
8. `y_zero_point` - output zero point
9. `B` - bias (INT32, optional)

**Two-Stage Computation:**

**Stage 1: ConvInteger (8×8→32 bit)**
```
Conv_result[n, c, h, w] = Σ Σ Σ (x[n, ic, ih+kh, iw+kw] - x_zero_point)
                              * (w[c, ic, kh, kw] - w_zero_point)
                          + B[c]

where the summation is over:
  ic: input channels (0 to in_channels/groups)
  kh: kernel height (0 to kernel_h)
  kw: kernel width (0 to kernel_w)

Result type: INT32 accumulator
```

**Stage 2: Requantization (32→8 bit)**
```
y[n, c, h, w] = saturate(
    round(Conv_result[n, c, h, w] * (x_scale * w_scale / y_scale))
    + y_zero_point
)

Saturation to INT8 [-128, 127] or UINT8 [0, 255]
```

**Bias Quantization Requirement:**
```
bias_scale = x_scale * w_scale
bias_zero_point = 0
bias_type = INT32
```

**Hardware Implementation:**
- 8-bit × 8-bit multipliers
- 32-bit accumulators (to prevent overflow during accumulation)
- Multiply-accumulate (MAC) units
- Scale multiplication and division
- Round-to-nearest logic
- Saturation to 8-bit output

**Convolution Attributes:**
- Kernel shape, strides, padding, dilation
- Groups (for depthwise/grouped convolutions)
- Auto-padding modes: NOTSET, SAME_UPPER, SAME_LOWER, VALID

**Per-Channel Weight Quantization:**
- w_scale: shape = (out_channels,)
- w_zero_point: shape = (out_channels,)
- Each output channel has independent quantization parameters

**ONNX Specification:** Version 10+
**Confidence:** HIGH (verified with official ONNX documentation and GitHub issues)

---

### 4. QLinearMatMul (Quantized Matrix Multiplication)

**Purpose:** Perform matrix multiplication entirely in quantized integer space

**High-Level Formula:**
```
QLinearMatMul = MatMulInteger + QuantizeLinear
```

**Inputs (8 total):**
1. `a` - quantized matrix A (INT8/UINT8)
2. `a_scale` - matrix A scale factor
3. `a_zero_point` - matrix A zero point
4. `b` - quantized matrix B (INT8/UINT8)
5. `b_scale` - matrix B scale factor
6. `b_zero_point` - matrix B zero point
7. `y_scale` - output scale factor
8. `y_zero_point` - output zero point

**Two-Stage Computation:**

**Stage 1: MatMulInteger (8×8→32 bit)**
```
MatMul_result[i, j] = Σ (a[i, k] - a_zero_point) * (b[k, j] - b_zero_point)
                      k=0 to K-1

where:
  a has shape (M, K)
  b has shape (K, N)
  result has shape (M, N)

Result type: INT32 accumulator
```

**Stage 2: Requantization (32→8 bit)**
```
y[i, j] = saturate(
    round(MatMul_result[i, j] * (a_scale * b_scale / y_scale))
    + y_zero_point
)

Saturation to INT8 [-128, 127] or UINT8 [0, 255]
```

**Hardware Implementation:**
- 8-bit × 8-bit multipliers
- 32-bit accumulators
- MAC (multiply-accumulate) units
- Scale multiplication and division
- Round-to-nearest-even logic
- Saturation to 8-bit output

**Per-Row/Column Quantization Support:**
- Supports per-tensor (scalar scale/zero_point)
- Supports per-row for A (scale shape: (M, 1))
- Supports per-column for B (scale shape: (1, N))

**Critical Constraint:**
"Production must never overflow, and accumulation may overflow if and only if in 32 bits"

**ONNX Specification:** Version 10+
**Confidence:** HIGH (verified with official ONNX documentation)

---

## ConvInteger and MatMulInteger (Lower-Level Operations)

These are the integer-only operations without built-in requantization.

### ConvInteger

**Output:** INT32 (not requantized)
**Operation:** 8×8→32 bit convolution
**Formula:**
```
y[n, c, h, w] = Σ Σ Σ (x[n, ic, ih+kh, iw+kw] - x_zero_point)
                    * (w[c, ic, kh, kw] - w_zero_point)
```

**Key Difference from QLinearConv:**
- ConvInteger outputs INT32 (dequantization needs separate step)
- QLinearConv outputs INT8/UINT8 (requantization built-in)
- Relationship: `QLinearConv = ConvInteger + QuantizeLinear`

**Use Case:**
Suitable when only certain operations need quantization, as INT32 output easily converts to FP32 for subsequent floating-point operations.

### MatMulInteger

**Output:** INT32 (not requantized)
**Operation:** 8×8→32 bit matrix multiplication
**Formula:**
```
y[i, j] = Σ (a[i, k] - a_zero_point) * (b[k, j] - b_zero_point)
          k
```

**Use Case:**
Similar to ConvInteger - useful when mixing quantized and non-quantized operations.

**Confidence:** HIGH (verified with ONNX documentation)

---

## PyTorch Quantized Operations

PyTorch provides quantized operations through `torch.ao.nn.quantized` module.

### torch.ao.nn.quantized.functional.conv2d

**Function Signature:**
```python
conv2d(input, weight, bias, stride=1, padding=0, dilation=1,
       groups=1, padding_mode='zeros', scale=1.0, zero_point=0,
       dtype=torch.quint8)
```

**Parameters:**
- `input`: Quantized input tensor (minibatch, in_channels, iH, iW)
- `weight`: Quantized filters (out_channels, in_channels/groups, kH, kW)
- `bias`: Non-quantized bias (FP32) - note the difference from ONNX INT32
- `scale`, `zero_point`: Output quantization parameters
- `dtype`: torch.quint8 or torch.qint8

**Mapping to ONNX:**
- PyTorch `torch.nn.quantized.Conv2d` → ONNX `QLinearConv`
- Export challenges exist (see Pitfalls section)

### torch.nn.quantized.functional.linear

**Function Signature:**
```python
linear(input, weight, bias=None, scale=None, zero_point=None)
```

**Parameters:**
- `input`: Quantized input tensor
- `weight`: Quantized weight tensor
- `bias`: Optional bias (FP32)
- `scale`, `zero_point`: Output quantization parameters

**Mapping to ONNX:**
- PyTorch `torch.nn.quantized.Linear` → ONNX `QLinearMatMul`
- Export challenges exist (see Pitfalls section)

**Performance Note:**
Current implementation packs weights on every call, which has performance penalty.

### torch.quantize_per_tensor

**Formula:**
```python
q = torch.quantize_per_tensor(x, scale, zero_point, dtype)

# Equivalent to ONNX QuantizeLinear
q[i] = saturate(round(x[i] / scale) + zero_point)
```

**Mapping to ONNX:**
- PyTorch `torch.quantize_per_tensor` → ONNX `QuantizeLinear`

### torch.dequantize / Tensor.dequantize()

**Formula:**
```python
y = q.dequantize()

# Equivalent to ONNX DequantizeLinear
y[i] = (q[i] - zero_point) * scale
```

**Mapping to ONNX:**
- PyTorch `.dequantize()` → ONNX `DequantizeLinear`

**Confidence:** MEDIUM (verified with PyTorch documentation, but export behavior is implementation-dependent)

---

## Per-Tensor vs Per-Channel Quantization

### Per-Tensor Quantization

**Characteristics:**
- Single scale and zero_point for entire tensor
- Scale and zero_point are scalars
- Same quantization parameters for all values

**Formula:**
```
q[...] = saturate(round(x[...] / scale) + zero_point)
```

**Use Cases:**
- Activations (feature maps)
- Simple quantization schemes
- Lower memory overhead

**PyTorch Type:** `torch.per_tensor_affine`

### Per-Channel Quantization

**Characteristics:**
- Different scale and zero_point for each channel
- Scale and zero_point are 1-D tensors (length = num_channels)
- Each channel quantized independently

**Formula:**
```
For convolution weights (out_channels, in_channels, kH, kW):

q[c, :, :, :] = saturate(
    round(x[c, :, :, :] / scale[c]) + zero_point[c]
)

where c iterates over output channels
```

**Use Cases:**
- Convolution weights (different channels have different ranges)
- Linear layer weights
- Better accuracy than per-tensor

**PyTorch Type:** `torch.per_channel_affine`

**Accuracy Impact:**
Per-channel quantization provides better accuracy because different channels (filters) often have different value distributions. Using channel-specific quantization parameters reduces quantization error.

**Hardware Implications:**
- Per-tensor: Single scale/zero_point register
- Per-channel: Array of scales/zero_points (size = num_channels)
- Memory overhead: Per-channel requires more parameter storage

**ONNX Support:**
- QLinearConv supports per-channel for weights
- QLinearMatMul supports per-row/per-column quantization
- Activations typically use per-tensor

**Confidence:** HIGH (verified with PyTorch and ONNX documentation)

---

## Hardware Implementation Requirements

### Arithmetic Units

| Operation | Input Precision | Accumulator | Output | Notes |
|-----------|----------------|-------------|--------|-------|
| MAC (multiply-accumulate) | 8-bit × 8-bit | 32-bit | 32-bit | Core computation unit |
| Requantization multiply | 32-bit × FP32 | FP32 or fixed-point | 32-bit | Scale multiplication |
| Division | 32-bit / FP32 | FP32 or fixed-point | 32-bit | Scale division |
| Saturation | 32-bit | - | 8-bit | Clipping to target range |

### Precision Requirements

**Convolution/MatMul Accumulation:**
- Input: 8-bit × 8-bit = 16-bit product
- Accumulation: Up to 32-bit (must not overflow for valid results)
- Example: For 3×3 kernel with 64 input channels = 576 accumulations
  - Max accumulation: 127 × 127 × 576 = 9,313,344 (fits in INT32)

**Requantization:**
- Scale multiplication: (x_scale × w_scale / y_scale)
- Can be precomputed as single scale factor: `M = (x_scale × w_scale) / y_scale`
- Hardware can use fixed-point arithmetic with bit-shifting

**Rounding:**
- Round-to-nearest-even (banker's rounding)
- Alternative: Use `nearbyintf()` or equivalent hardware rounding mode

### Register/Memory Requirements

**Per-Tensor Quantization:**
- 2 values per tensor: scale (FP32), zero_point (INT8/UINT8)

**Per-Channel Quantization (for Conv2D with 64 output channels):**
- 64 scales (FP32 × 64 = 256 bytes)
- 64 zero_points (INT8 × 64 = 64 bytes)
- Total: 320 bytes per layer

**Bias:**
- INT32 per output channel
- For 64 channels: 256 bytes

### Optimization Opportunities

**Fixed-Point Requantization:**
```
Instead of: result * (x_scale * w_scale / y_scale)

Use: (result * M) >> N

where:
  M = round((x_scale * w_scale / y_scale) * 2^N)
  N = bit shift amount (typically 8-16)
```

This avoids floating-point operations in hardware.

**Recent Research (2025-2026):**
- Bit-shifting techniques achieve ~27% FPS improvement
- RISC-V implementations reach 13 GOPS (8-bit) and 23 GOPS (4-bit)
- Energy efficiency: 270 GOPS/W (8-bit), 405 GOPS/W (4-bit)

**Confidence:** MEDIUM (based on recent research papers and hardware implementation studies)

---

## ResNet8-Specific Operations

Based on the ResNet8 architecture with Conv2D layers (16, 32, 64 filters) and Dense layer (10 outputs):

### Layer-by-Layer Quantization Parameters

| Layer | Type | Weights | Bias | Activations | Notes |
|-------|------|---------|------|-------------|-------|
| Conv2D-1 | QLinearConv | Per-channel (16) | INT32 × 16 | Per-tensor | Input layer |
| Conv2D-2 | QLinearConv | Per-channel (32) | INT32 × 32 | Per-tensor | Mid layer |
| Conv2D-3 | QLinearConv | Per-channel (64) | INT32 × 64 | Per-tensor | Mid layer |
| Dense | QLinearMatMul | Per-channel (10) | INT32 × 10 | Per-tensor | Output layer |

### BatchNorm and ReLU

**BatchNorm Quantization:**
BatchNorm in quantized networks is typically fused into the preceding convolution:

```
Fused operation:
y = Conv(x, w_fused, b_fused)

where:
  w_fused = w * (gamma / sqrt(var + eps))
  b_fused = beta + (b - mean) * (gamma / sqrt(var + eps))

This avoids quantizing BatchNorm separately.
```

**ReLU Quantization:**
ReLU in quantized space is a simple comparison:

```
ReLU(x_quantized):
  if x_quantized < zero_point:
    return zero_point
  else:
    return x_quantized
```

No dequantization needed - operates directly on quantized values.

**AvgPool Quantization:**
Average pooling requires careful handling:

```
Step 1: Sum quantized values (result in INT32)
Step 2: Divide by pool size
Step 3: Requantize to output scale/zero_point

Formula:
avg = (sum(x_quantized) - pool_size * x_zero_point) / pool_size
y_quantized = round(avg * x_scale / y_scale) + y_zero_point
```

**Confidence:** HIGH (standard quantization practices)

---

## Differentiators: Advanced Features

Features that provide more flexibility and optimization for hardware implementations.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Fixed-point requantization** | Avoid FP ops, use bit-shifting | Medium | 27% FPS improvement reported |
| **Per-channel quantization** | Better accuracy for weights | Low | Industry standard for Conv/Linear |
| **Fused operations** | Reduce memory bandwidth | Medium | Conv+BN fusion, Conv+ReLU fusion |
| **Symmetric vs asymmetric** | Simpler hardware for symmetric | Low | Symmetric weights, asymmetric activations |
| **INT4 quantization** | 2× memory reduction vs INT8 | High | Emerging, lower accuracy |
| **Mixed-precision** | Critical layers in higher precision | High | Requires sensitivity analysis |

---

## Anti-Features: Operations to Avoid

### 1. Mixed Precision without Explicit Conversion

**What NOT to do:**
Mixing quantized and floating-point operations without explicit QuantizeLinear/DequantizeLinear.

**Why it's bad:**
Creates undefined behavior and prevents hardware optimization.

**Instead:**
Always use explicit QuantizeLinear/DequantizeLinear at boundaries:
```
FP32 → QuantizeLinear → INT8 operations → DequantizeLinear → FP32
```

### 2. Dynamic Quantization for Hardware Accelerators

**What NOT to do:**
Using dynamic quantization where scales are computed at runtime per-batch.

**Why it's bad:**
- Hardware accelerators need fixed quantization parameters
- Runtime scale computation defeats purpose of integer-only inference
- Cannot precompute requantization factors

**Instead:**
Use static quantization with calibration to determine fixed scales.

### 3. FP16 as "Quantization"

**What NOT to do:**
Calling FP16 inference "quantization" for hardware accelerators.

**Why it's bad:**
- Requires floating-point hardware
- Misses 4× memory reduction of INT8
- Not true quantization in the integer sense

**Instead:**
Use INT8 quantization for edge/hardware deployment.

### 4. Per-Tensor Weights in Convolution

**What NOT to do:**
Using per-tensor quantization for convolution weights.

**Why it's bad:**
- Significant accuracy loss (especially in deep networks)
- Different channels have different ranges
- Industry standard is per-channel for weights

**Instead:**
Always use per-channel quantization for convolution and linear layer weights.

### 5. Ignoring Bias Quantization Requirements

**What NOT to do:**
Quantizing bias independently or using INT8 for bias.

**Why it's bad:**
- ONNX spec requires: `bias_scale = input_scale × weight_scale`
- Bias must be INT32 to match accumulator precision
- Wrong bias quantization breaks numerical accuracy

**Instead:**
Follow ONNX spec: INT32 bias with `scale = x_scale × w_scale`, `zero_point = 0`.

---

## Feature Dependencies

```
QuantizeLinear
  └─> Quantized Tensors (INT8/UINT8)
       ├─> QLinearConv
       │    ├─> ConvInteger (Stage 1: 8×8→32)
       │    └─> Requantization (Stage 2: 32→8)
       ├─> QLinearMatMul
       │    ├─> MatMulInteger (Stage 1: 8×8→32)
       │    └─> Requantization (Stage 2: 32→8)
       └─> DequantizeLinear
            └─> FP32 Output

Hardware Implementation:
  ├─> 8×8 MAC Units
  ├─> 32-bit Accumulators
  ├─> Scale/Zero-Point Registers
  ├─> Rounding Logic
  └─> Saturation Logic
```

---

## Sources

### ONNX Official Documentation
- [QLinearConv - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearConv.html)
- [QLinearMatMul - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html)
- [QuantizeLinear - ONNX 1.21.0 documentation](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [DequantizeLinear - ONNX 1.21.0 documentation](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)
- [ConvInteger - ONNX 1.21.0 documentation](https://onnx.ai/onnx/operators/onnx__ConvInteger.html)
- [Quantize ONNX models | onnxruntime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### ONNX GitHub Issues
- [ConvInteger vs QLinearConv · Issue #2424 · onnx/onnx](https://github.com/onnx/onnx/issues/2424)
- [How is QlinearConv calculated? · Issue #11883 · microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/issues/11883)
- [Prevent int32 quantized bias from clipping · Pull Request #22020 · microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/pull/22020)

### PyTorch Documentation
- [Quantization Overview — torchao 0.15 documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html)
- [quantize_per_tensor — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)
- [conv2d — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.ao.nn.quantized.functional.conv2d.html)
- [Quantization API Reference — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/quantization-support.html)

### Research Papers (2025-2026)
- [Quantized convolutional neural networks: a hardware perspective - Frontiers](https://www.frontiersin.org/journals/electronics/articles/10.3389/felec.2025.1469802/full)
- [Speed up integer-arithmetic-only inference via bit-shifting - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-02544-4)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference - arXiv](https://ar5iv.labs.arxiv.org/html/1712.05877)

---

## Confidence Assessment

| Operation | Confidence | Source |
|-----------|-----------|--------|
| QuantizeLinear formula | HIGH | ONNX official spec v1.21 |
| DequantizeLinear formula | HIGH | ONNX official spec v1.21 |
| QLinearConv formula | HIGH | ONNX spec + GitHub issue #11883 |
| QLinearMatMul formula | HIGH | ONNX spec + GitHub issue #2424 |
| PyTorch operations | MEDIUM | PyTorch 2.10 docs (export behavior varies) |
| Hardware implementation | MEDIUM | Research papers + ONNX spec constraints |
| Per-tensor vs per-channel | HIGH | ONNX spec + PyTorch docs |
| ResNet8 specifics | HIGH | Standard quantization practices |

---

## Next Steps for Documentation

1. **Create detailed examples** showing QLinearConv calculation step-by-step for a simple 3×3 convolution
2. **Document calibration process** for determining scale/zero_point from FP32 model
3. **Hardware architecture diagrams** showing dataflow through MAC units
4. **Numerical precision analysis** for different bit-widths and accumulator sizes
5. **ONNX export workflow** from PyTorch quantized model to ONNX QLinear operators
6. **Testing methodology** for verifying hardware implementation correctness

**Readiness:** Research complete. Ready for requirements definition and detailed specification writing.
