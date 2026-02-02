# Phase 11: Core Operations Documentation — Research

**Phase:** QLinearConv and QLinearMatMul operations fully documented with two-stage computation explained

**Research Date:** 2026-02-02

**Overall Confidence:** HIGH

## Executive Summary

Quantized neural network operations (QLinearConv, QLinearMatMul) are well-established in ONNX with official specifications and reference implementations. The key insight for documenting these operations is that the two-stage computation pattern (INT8×INT8→INT32 MAC, then requantization to INT8) is universal and must be made explicit. Per-channel quantization adds complexity through vector scales but follows the same fundamental pattern.

**Documentation approach:** Start with authoritative ONNX operator specifications, show raw arithmetic using NumPy (avoiding high-level APIs), provide worked examples with actual ResNet8 values, and explicitly handle edge cases (overflow, saturation, rounding).

**Critical insight for analog accelerator implementers:** INT32 accumulator is non-negotiable. Attempting lower bitwidth accumulators introduces numerical errors that degrade accuracy. The two-stage computation pattern separates concerns: stage 1 performs integer arithmetic (hardware-friendly), stage 2 applies scaling/requantization (requires higher precision).

## Standard Stack

**MANDATORY: Use these technologies**

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| ONNX | 1.20+ | Operator specification | Official standard, authoritative source for QLinearConv/QLinearMatMul specifications |
| NumPy | Latest | Arithmetic examples | Raw arithmetic operations without abstraction, matches hardware implementation closely |
| Python | 3.8+ | Pseudocode language | Runnable examples, widely understood, good for visualization |
| Matplotlib | Latest | Visualization | Plots for quantization ranges, distribution analysis |
| ONNX Runtime | Latest | Reference implementation | Official Python reference evaluator for validation |

**AVOID: Do NOT use these**

| Technology | Why Not |
|------------|---------|
| torch.quantization API | High-level abstractions hide the raw arithmetic we need to explain |
| torch.ao.quantization | Same issue — abstracts away bit-level operations |
| TensorFlow Lite quantization | Framework-specific, not hardware-agnostic |
| Custom quantization libraries | Not standard, introduces unnecessary dependencies |

**Rationale:** The target audience (analog accelerator implementers) needs to see raw quantization arithmetic, not framework abstractions. NumPy operations directly map to hardware operations: `(x / scale).round().clip(-128, 127)` is exactly what hardware implements.

## Architecture Patterns

### Pattern 1: Two-Stage Quantized Computation (Universal)

**What:** Separate integer arithmetic from scaling operations

**When:** All quantized convolution and matmul operations

**Structure:**
```python
# Stage 1: Integer MAC operations
# Input: INT8, Weights: INT8
# Accumulator: INT32 (REQUIRED to prevent overflow)
acc = np.int32(0)
for i in range(K):
    acc += np.int32(x_int8[i]) * np.int32(w_int8[i])

# Stage 2: Requantization to output
# Apply all scale factors at once
scale_factor = (x_scale * w_scale) / y_scale
result_float = (acc - intermediate_zero_point) * scale_factor
result_int8 = np.round(result_float / y_scale + y_zero_point).clip(-128, 127).astype(np.int8)
```

**Why this pattern:**
- Separates integer operations (hardware-efficient) from floating-point scaling (precision-critical)
- INT32 accumulator prevents overflow during MAC operations
- All scale factors applied in single pass (numerically stable)
- Maps directly to hardware pipeline stages

**Anti-pattern to avoid:**
```python
# WRONG: Applying scales during accumulation
for i in range(K):
    # This causes precision loss and doesn't match hardware
    acc += (x_int8[i] * x_scale) * (w_int8[i] * w_scale)
```

### Pattern 2: Per-Channel Quantization (Weight-side)

**What:** Different scale/zero-point per output channel

**When:** Convolution weights (common), matmul weights (less common)

**Structure:**
```python
# w_scale shape: [num_output_channels]
# w_zero_point shape: [num_output_channels]

for c in range(num_output_channels):
    # Each channel has its own quantization parameters
    channel_scale = w_scale[c]
    channel_zero_point = w_zero_point[c]

    # Stage 1: MAC with INT8 values
    acc = compute_mac_int32(x_int8, w_int8[c, :, :, :])

    # Stage 2: Requantization with channel-specific scale
    scale_factor = (x_scale * channel_scale) / y_scale
    output[c] = requantize(acc, scale_factor, y_zero_point)
```

**Why per-channel:**
- Different output channels often have vastly different weight magnitudes
- Per-tensor quantization would waste dynamic range on small-magnitude channels
- Storage cost is minimal (one scale + zero-point per channel, not per weight)

**Memory implications:**
- Per-tensor: 2 parameters (1 scale, 1 zero-point)
- Per-channel (256 channels): 512 parameters (256 scales, 256 zero-points)
- Weight tensor (256×128×3×3): 294,912 parameters
- Overhead: 512/294,912 = 0.17% — negligible

### Pattern 3: Rounding Mode (Ties-to-Even)

**What:** IEEE 754 round-half-to-even (banker's rounding)

**When:** All quantization and requantization steps

**Implementation:**
```python
# NumPy default is round-half-to-even (correct)
quantized = np.round(float_value / scale + zero_point)

# NOT: Python's built-in round() which is round-half-away-from-zero
# WRONG: quantized = round(float_value / scale + zero_point)
```

**Why ties-to-even:**
- ONNX specification requirement
- Eliminates statistical bias (equal probability of rounding up/down)
- Hardware implementations use this (IEEE 754 default)

**Edge case example:**
```python
# Value exactly halfway between integers
value = 2.5
# Ties-to-even: 2.5 → 2 (even)
# 3.5 → 4 (even)
# Eliminates upward bias over many rounding operations
```

### Pattern 4: Saturation Arithmetic (Clipping)

**What:** Clamp values to valid INT8 range [-128, 127] after rounding

**When:** After every quantization and requantization operation

**Implementation:**
```python
# REQUIRED: Saturation, not wrapping
result = np.round(value).clip(-128, 127).astype(np.int8)

# WRONG: Wrapping behavior (C-style overflow)
# result = np.int8(np.round(value))  # 130 becomes -126 (wraps)
```

**Why saturation:**
- Gradual degradation vs catastrophic failure
- Matches neural network training behavior (activations naturally clipped)
- Hardware implementations typically provide saturation instructions

**Hardware note:** Most ISAs provide saturating arithmetic instructions (e.g., x86 PADDSB, ARM SQADD). Document this for implementers.

## Documentation Structure (Standard Pattern)

Based on ONNX documentation style and quantization white papers, structure documentation as:

### For Each Operator (QLinearConv, QLinearMatMul)

1. **Overview** (3-4 sentences)
   - What the operator does (one sentence)
   - Relationship to standard Conv/MatMul (one sentence)
   - Key differences from FP32 version (one sentence)

2. **Input Specification**
   - Table with columns: Name | Type | Shape | Description
   - All 8-9 inputs for QLinearConv
   - All 8 inputs for QLinearMatMul
   - Note per-tensor vs per-channel options

3. **Computation Formula**
   - Mathematical notation (brief)
   - Then immediate expansion to two-stage process
   - Emphasize INT32 accumulator requirement

4. **Worked Example: Per-Tensor Quantization**
   - Use first QLinearConv from ResNet8
   - Show actual INT8 values from ONNX model
   - Step through Stage 1 (MAC) with explicit INT32 accumulator
   - Step through Stage 2 (requantization) with scale factors
   - Include Python code that verifies against ONNX Runtime output

5. **Worked Example: Per-Channel Quantization**
   - Use later ResNet8 layer with per-channel weights
   - Show scale array [c0, c1, c2, ...]
   - Compute 2-3 output channels explicitly
   - Show how requantization differs per channel

6. **Edge Cases**
   - Overflow handling: Why INT32 accumulator is required
   - Saturation: What happens when requantization exceeds [-128, 127]
   - Rounding: Ties-to-even examples
   - Zero-point handling: Subtracting zero-points before MAC

7. **Hardware Implementation Pseudocode**
   - Runnable Python with explicit type annotations
   - Comments mapping to hardware stages
   - Visualization code for input/output distributions

8. **Validation Snippet**
   - Load ONNX model
   - Extract scale/zero-point values
   - Run manual computation
   - Compare with ONNX Runtime output
   - Assert tolerance (exact match for INT8)

## Code Examples (Reference Pattern)

### Example 1: Manual Per-Tensor QLinearConv

```python
import numpy as np

def qlinear_conv_manual(
    x: np.ndarray,  # INT8 input [N, C, H, W]
    x_scale: float,
    x_zero_point: int,  # INT8
    w: np.ndarray,  # INT8 weights [M, C, kH, kW]
    w_scale: float,  # Per-tensor: scalar
    w_zero_point: int,  # INT8
    y_scale: float,
    y_zero_point: int,  # INT8
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:  # INT8 output [N, M, H_out, W_out]
    """
    QLinearConv with per-tensor quantization.

    Implements two-stage computation:
    Stage 1: INT8×INT8→INT32 MAC operations
    Stage 2: INT32→INT8 requantization with scaling
    """
    N, C_in, H, W = x.shape
    M, C_in, kH, kW = w.shape

    # Calculate output dimensions
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1

    # Apply padding if needed
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)),
                   mode='constant', constant_values=x_zero_point)

    # Output accumulator: INT32 to prevent overflow
    output_int32 = np.zeros((N, M, H_out, W_out), dtype=np.int32)

    # Stage 1: Integer MAC operations
    for n in range(N):
        for m in range(M):  # Output channels
            for h in range(H_out):
                for w_out in range(W_out):
                    # Extract patch
                    h_start = h * stride
                    w_start = w_out * stride
                    patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]

                    # MAC operation in INT32
                    # Subtract zero-points BEFORE multiplication
                    x_dequant = patch.astype(np.int32) - x_zero_point
                    w_dequant = w[m].astype(np.int32) - w_zero_point

                    # Accumulate in INT32
                    output_int32[n, m, h, w_out] = np.sum(x_dequant * w_dequant)

    # Stage 2: Requantization to INT8
    # Combined scale factor
    scale_factor = (x_scale * w_scale) / y_scale

    # Apply scale and requantize
    output_float = output_int32.astype(np.float32) * scale_factor
    output_quantized = np.round(output_float / y_scale + y_zero_point)

    # Saturate to INT8 range
    output_int8 = np.clip(output_quantized, -128, 127).astype(np.int8)

    return output_int8
```

**Usage pattern:**
```python
# Load actual ResNet8 values from ONNX model
x_int8 = load_from_onnx("input_quantized")
w_int8 = load_from_onnx("conv1.weight_quantized")
x_scale, x_zero_point = load_from_onnx("input_scale"), load_from_onnx("input_zero_point")
# ... load all 8 parameters

result = qlinear_conv_manual(x_int8, x_scale, x_zero_point, w_int8, w_scale, w_zero_point, y_scale, y_zero_point)

# Validate against ONNX Runtime
onnx_result = run_onnx_runtime("QLinearConv_node_1")
assert np.allclose(result, onnx_result), "Implementation matches ONNX Runtime"
```

### Example 2: Per-Channel Extension

```python
def qlinear_conv_per_channel(
    x: np.ndarray,
    x_scale: float,
    x_zero_point: int,
    w: np.ndarray,
    w_scale: np.ndarray,  # Changed: [M] array, one per output channel
    w_zero_point: np.ndarray,  # Changed: [M] array
    y_scale: float,
    y_zero_point: int,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """Per-channel quantization: different scale/zero-point per output channel."""

    # ... same setup as per-tensor ...

    # Stage 1: Identical to per-tensor (MAC operations unchanged)
    # ... compute output_int32 ...

    # Stage 2: Requantization differs per channel
    M = w.shape[0]  # Number of output channels
    output_int8 = np.zeros_like(output_int32, dtype=np.int8)

    for m in range(M):
        # Each channel has its own scale factor
        channel_scale_factor = (x_scale * w_scale[m]) / y_scale

        # Apply channel-specific scale
        channel_float = output_int32[:, m, :, :].astype(np.float32) * channel_scale_factor
        channel_quantized = np.round(channel_float / y_scale + y_zero_point)

        # Saturate
        output_int8[:, m, :, :] = np.clip(channel_quantized, -128, 127).astype(np.int8)

    return output_int8
```

**Key difference:** Only Stage 2 changes. Stage 1 (MAC operations) identical between per-tensor and per-channel.

### Example 3: Overflow Demonstration

```python
def demonstrate_accumulator_overflow():
    """
    Show why INT32 accumulator is required.

    For 3×3 convolution with 128 input channels:
    - 128 channels × 3×3 kernel = 1,152 MAC operations
    - Worst case: 127 × 127 × 1,152 = 18,579,456
    - INT16 max: 32,767 — OVERFLOWS!
    - INT32 max: 2,147,483,647 — safe
    """
    # Worst-case input: all 127 (max INT8 positive value)
    C, kH, kW = 128, 3, 3
    x_patch = np.full((C, kH, kW), 127, dtype=np.int8)
    w_kernel = np.full((C, kH, kW), 127, dtype=np.int8)

    # Try INT16 accumulator (WRONG)
    acc_int16 = np.int16(0)
    for val_x, val_w in zip(x_patch.flat, w_kernel.flat):
        acc_int16 += np.int16(val_x) * np.int16(val_w)
        # Overflows silently!

    print(f"INT16 accumulator result: {acc_int16}")  # Wrong due to overflow

    # Correct: INT32 accumulator
    acc_int32 = np.int32(0)
    for val_x, val_w in zip(x_patch.flat, w_kernel.flat):
        acc_int32 += np.int32(val_x) * np.int32(val_w)

    print(f"INT32 accumulator result: {acc_int32}")  # Correct: 18,579,456
    print(f"INT16 would overflow by: {acc_int32 / 32767:.1f}× the INT16 max")
```

## Don't Hand-Roll

**NEVER implement these from scratch — use authoritative sources:**

1. **ONNX Operator Specifications**
   - Source: https://onnx.ai/onnx/operators/
   - Use official input/output specifications exactly
   - Don't interpret or guess parameter meanings

2. **Rounding Mode Implementation**
   - Use NumPy's np.round() (implements ties-to-even correctly)
   - Don't implement custom rounding functions
   - Don't use Python's built-in round() (wrong behavior)

3. **ONNX Model Loading/Parsing**
   - Use onnx.load() and onnx.helper APIs
   - Don't parse .onnx files manually
   - Use ONNX Runtime's ReferenceEvaluator for validation

4. **Quantization Parameter Extraction**
   - Use ONNX Runtime APIs to extract scales/zero-points
   - Don't hard-code quantization parameters
   - Verify extracted values against ONNX metadata

5. **Convolution Operation Itself**
   - Per CONTEXT.md: "Skip — only document what quantization changes"
   - Assume reader knows convolution mechanics
   - Focus only on quantization-specific aspects

## Common Pitfalls

### Pitfall 1: Using Too-Small Accumulator

**What goes wrong:** INT16 or smaller accumulator overflows during MAC operations, producing wrong results

**Why it happens:** Intuition says "INT8 × INT8 = INT16 is enough"

**Consequences:**
- Silent numerical errors (overflow wraps)
- Accuracy drops significantly
- Errors accumulate across layers

**Prevention:**
```python
# WRONG: INT16 accumulator
acc = np.int16(0)
for x_val, w_val in zip(x_int8, w_int8):
    acc += np.int16(x_val) * np.int16(w_val)  # Overflows!

# CORRECT: INT32 accumulator
acc = np.int32(0)
for x_val, w_val in zip(x_int8, w_int8):
    acc += np.int32(x_val) * np.int32(w_val)  # Safe
```

**Detection:** Compare manual implementation against ONNX Runtime. If results differ significantly (>10 INT8 units), likely accumulator overflow.

**Confidence:** HIGH — This is explicitly stated in ONNX QLinearMatMul specification and discussed in multiple quantization papers

### Pitfall 2: Forgetting Zero-Point Subtraction

**What goes wrong:** Treating quantized INT8 values as if zero-point is always 0

**Why it happens:** Per-tensor symmetric quantization (zero-point = 0) is common, leading to assumption it's always 0

**Consequences:**
- Asymmetric quantization produces wrong results
- Bias term computed incorrectly

**Prevention:**
```python
# WRONG: Ignoring zero-points
acc = np.int32(x_int8[i]) * np.int32(w_int8[i])

# CORRECT: Subtract zero-points before MAC
x_dequant = np.int32(x_int8[i]) - x_zero_point
w_dequant = np.int32(w_int8[i]) - w_zero_point
acc = x_dequant * w_dequant
```

**Detection:** Test with asymmetric quantization (non-zero zero-points). If results are wrong, likely missing zero-point handling.

**Confidence:** HIGH — ONNX specification explicitly shows zero-point subtraction in computation formula

### Pitfall 3: Wrong Rounding Mode

**What goes wrong:** Using round-half-up instead of round-half-to-even

**Why it happens:** Python's built-in round() uses ties-away-from-zero (different from NumPy)

**Consequences:**
- Statistical bias (always rounds up on ties)
- Accumulates over many operations
- Results don't match ONNX Runtime

**Prevention:**
```python
# WRONG: Python's round()
quantized = int(round(value))  # Uses ties-away-from-zero

# CORRECT: NumPy's round()
quantized = np.round(value)  # Uses ties-to-even (IEEE 754)
```

**Detection:** Create test case with exact halfway values (e.g., 2.5, 3.5). NumPy rounds to even, Python rounds away from zero.

**Confidence:** HIGH — ONNX specification states "rounding uses nearest ties-to-even method"

### Pitfall 4: Applying Scales Too Early

**What goes wrong:** Converting to float and applying scales during MAC operations

**Why it happens:** Intuition from floating-point neural networks

**Consequences:**
- Precision loss during accumulation
- Doesn't match hardware implementation pattern
- Much slower (float operations vs integer)

**Prevention:**
```python
# WRONG: Scales during accumulation
acc_float = 0.0
for i in range(K):
    acc_float += (x_int8[i] * x_scale) * (w_int8[i] * w_scale)

# CORRECT: Scales after accumulation
acc_int32 = np.int32(0)
for i in range(K):
    acc_int32 += np.int32(x_int8[i]) * np.int32(w_int8[i])
# Then apply scales once
result_float = acc_int32 * (x_scale * w_scale)
```

**Confidence:** HIGH — Fundamental to quantized inference, stated in all quantization literature

### Pitfall 5: Saturation vs Wrapping Confusion

**What goes wrong:** Allowing integer overflow to wrap instead of saturating

**Why it happens:** C/C++ default integer behavior is wrapping

**Consequences:**
- Large values become large negative values (130 → -126)
- Completely wrong output
- Hard to debug (looks like random corruption)

**Prevention:**
```python
# WRONG: Wrapping overflow
result = np.int8(value)  # 130 becomes -126

# CORRECT: Saturation
result = np.clip(value, -128, 127).astype(np.int8)  # 130 becomes 127
```

**Detection:** Test with values known to exceed [-128, 127]. Check if output saturates or wraps.

**Confidence:** HIGH — Neural network training uses saturation; wrapping breaks learned quantization parameters

### Pitfall 6: Per-Channel Scale Indexing Error

**What goes wrong:** Indexing per-channel scales by wrong dimension

**Why it happens:** Confusion about which dimension is "channel" (input vs output channel)

**Consequences:**
- Wrong scale applied to wrong channel
- Model produces garbage output
- May not be caught by shape checks

**Prevention:**
```python
# WRONG: Indexing by input channel
for c_in in range(C_in):
    scale = w_scale[c_in]  # Wrong dimension!

# CORRECT: Indexing by output channel
for c_out in range(C_out):
    scale = w_scale[c_out]  # Per output channel
```

**Detection:** Per-channel scales array should have shape [num_output_channels], not [num_input_channels]. Verify with ONNX model inspection.

**Confidence:** HIGH — ONNX specification states per-channel quantization is "per output channel"

### Pitfall 7: Confusing Quantization Order

**What goes wrong:** Thinking quantization parameters are just metadata, not actual data inputs

**Why it happens:** In FP32 models, scale factors aren't explicit inputs

**Consequences:**
- Trying to "configure" scale/zero-point instead of reading from model
- Forgetting to pass all 8-9 inputs to QLinearConv

**Prevention:**
- QLinearConv has 8-9 inputs (not 1 input + 7-8 attributes)
- All scale/zero-point values are data inputs, not attributes
- Must load from ONNX model explicitly

**Confidence:** HIGH — ONNX operator signature clearly shows 8-9 inputs

## Phase-Specific Warnings

| Topic | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| QLinearConv documentation | Assuming 9 inputs always required | Clarify: 9th input (bias) is optional |
| Per-channel examples | Using wrong ResNet8 layer (some are per-tensor) | Verify layer quantization type before using in example |
| Overflow examples | Not showing actual overflow (just theoretical calculation) | Include runnable code that triggers overflow with INT16, succeeds with INT32 |
| Validation snippets | Comparing float values with == | Use np.allclose() for floating-point comparisons, exact match for INT8 |
| Hardware pseudocode | Too abstract (high-level) | Show explicit type casts, bit-widths in comments |
| Rounding edge cases | Only showing 2.5 example | Show both 2.5 → 2 and 3.5 → 4 (demonstrates ties-to-even pattern) |
| Zero-point handling | Unclear when to subtract zero-point | State explicitly: subtract before MAC, add after requantization |

## Verification Checklist

Before considering documentation complete:

**Correctness Verification:**
- [ ] All QLinearConv inputs documented (8 required, 1 optional)
- [ ] All QLinearMatMul inputs documented (8 required)
- [ ] Two-stage computation shown explicitly in pseudocode
- [ ] INT32 accumulator requirement stated with overflow example
- [ ] Rounding mode (ties-to-even) demonstrated with code
- [ ] Saturation behavior shown with clipping example
- [ ] Zero-point handling shown in both MAC and requantization stages

**Example Verification:**
- [ ] At least one per-tensor example using actual ResNet8 ONNX values
- [ ] At least one per-channel example using actual ResNet8 ONNX values
- [ ] Validation code runs and matches ONNX Runtime output
- [ ] All examples use raw NumPy arithmetic (not torch.quantization APIs)
- [ ] Examples include visualization (matplotlib plots of distributions)

**Edge Case Coverage:**
- [ ] Overflow scenario demonstrated (INT16 fails, INT32 succeeds)
- [ ] Saturation scenario demonstrated (value >127 clips to 127)
- [ ] Ties-to-even rounding demonstrated (2.5→2, 3.5→4)
- [ ] Asymmetric quantization demonstrated (non-zero zero-points)

**Hardware Implementer Needs:**
- [ ] Pseudocode specifies INT32 accumulator explicitly
- [ ] Stage 1 and Stage 2 clearly separated
- [ ] Memory layout considerations noted (per-channel scale storage)
- [ ] Practical implications documented (why INT32, why saturation)

**Anti-Pattern Detection:**
- [ ] Warning about INT16 accumulator included
- [ ] Warning about wrong rounding mode included
- [ ] Warning about premature scale application included
- [ ] Warning about wrapping vs saturation included

## Sources

### Official Specifications (HIGH Confidence)
- [QLinearConv - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearConv.html)
- [QLinearMatMul - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html)
- [Quantize ONNX models | onnxruntime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### Quantization Best Practices (HIGH Confidence)
- [What Is int8 Quantization and Why Is It Popular for Deep Neural Networks? - MATLAB & Simulink](https://www.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html)
- [Keras documentation: 8-bit Integer Quantization in Keras](https://keras.io/guides/int8_quantization_in_keras/)
- [Intel Neural Compressor documentation](https://intel.github.io/neural-compressor/2.0/quantization.html)

### Academic References (HIGH Confidence)
- [INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE - arXiv](https://arxiv.org/pdf/2004.09602)
- [A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295)
- [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342)

### Hardware Implementation (MEDIUM-HIGH Confidence)
- [Frontiers | Quantized convolutional neural networks: a hardware perspective](https://www.frontiersin.org/journals/electronics/articles/10.3389/felec.2025.1469802/full)
- [A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance - ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Colbert_A2Q_Accumulator-Aware_Quantization_with_Guaranteed_Overflow_Avoidance_ICCV_2023_paper.pdf)
- [NVDLA Hardware Architectural Specification](https://nvdla.org/hw/v1/hwarch.html)

### Visualization and Teaching (MEDIUM Confidence)
- [A Visual Guide to Quantization - by Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [Quantization in LLMs, Block-Wise Quantization with Realistic Data and Visualization — Part 3](https://medium.com/@kunwarmahen/quantization-in-llms-block-wise-quantization-with-realistic-data-and-visualization-part-3-13eeadc7e3d5)

### ONNX Debugging Tools (MEDIUM Confidence)
- [ONNX Runtime Quantization Debugging](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md)
- [AMD Quark ONNX Profiling](https://quark.docs.amd.com/release-0.11/onnx/tutorial_profiling.html)

### PyTorch Reference (for anti-patterns)
- [PyTorch Quantization API](https://docs.pytorch.org/docs/stable/quantization.html) — Document what NOT to use
- [A Manual Implementation of Quantization in PyTorch](https://franciscormendes.github.io/2024/05/16/quantization-1/) — Shows raw arithmetic approach

## Confidence Assessment

| Area | Confidence | Rationale |
|------|-----------|-----------|
| Two-stage computation pattern | HIGH | Explicit in ONNX spec, universal in all quantization literature |
| INT32 accumulator requirement | HIGH | Stated in ONNX QLinearMatMul spec, proven in accumulator overflow papers |
| Rounding mode (ties-to-even) | HIGH | Explicitly stated in ONNX specification |
| Saturation vs wrapping | HIGH | Standard neural network quantization practice, NVIDIA TensorRT docs confirm |
| Per-channel indexing (output channel) | HIGH | ONNX specification explicitly states "per output channel" |
| NumPy as reference implementation | HIGH | ONNX reference implementation uses NumPy, widely used in tutorials |
| Visualization approach | MEDIUM | Multiple sources show visualization, but no single standard |
| ResNet8 layer selection | MEDIUM | Need to inspect actual ONNX model to confirm which layers are per-channel |

## Open Questions / Gaps

**Resolved during research:**
- ✓ Which rounding mode? → Ties-to-even (ONNX spec)
- ✓ Accumulator bitwidth? → INT32 required (ONNX spec + overflow analysis)
- ✓ Per-channel dimension? → Output channel (ONNX spec)
- ✓ Python vs pseudocode? → Python with NumPy (Context.md decision)

**To investigate during implementation:**
- Which specific ResNet8 layers use per-channel vs per-tensor quantization?
- What are the actual scale/zero-point values in the exported ONNX model?
- Does ResNet8 ONNX model include bias term (9th input)?

**Deferred to later phases:**
- Advanced optimization techniques (Winograd, FFT convolution) — not relevant for analog accelerator learning
- Quantization-aware training details — Phase 11 is inference-only
- Mixed-precision quantization — ResNet8 is uniform INT8

## Ready for Planning

**Verdict:** READY

All major architecture patterns identified, standard stack determined, pitfalls catalogued. Documentation structure follows established ONNX patterns. Code examples ready to implement.

**Next step:** `/gsd:plan-phase 11` can use this research to structure tasks for creating documentation with worked examples, validation code, and edge case coverage.

**Key insight for planner:** The main documentation effort is NOT explaining convolution (skip that) but rather showing the quantization-specific arithmetic: zero-point subtraction, INT32 accumulation, scale factor application, and requantization. Use actual ResNet8 values throughout to keep examples concrete.
