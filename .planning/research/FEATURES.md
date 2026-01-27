# Feature Landscape: PTQ Evaluation

**Domain:** Post-Training Quantization (PTQ) for ResNet8 CIFAR-10
**Researched:** 2026-01-28
**Context:** Adding PTQ evaluation to existing full-precision evaluation codebase (87.19% baseline accuracy)

## Table Stakes

Features users expect for PTQ evaluation. Missing = incomplete PTQ assessment.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **ONNX Runtime static quantization** | Industry-standard quantization path for ONNX models | Medium | Uses `quantize_static()` API with calibration data |
| **PyTorch static quantization** | Native PyTorch quantization for converted models | Medium | New PT2E export-based approach recommended (88% model coverage) |
| **int8 quantization support** | Standard 8-bit signed quantization, most common format | Low | Symmetric for weights, asymmetric for activations |
| **uint8 quantization support** | Unsigned 8-bit quantization, hardware-dependent benefits | Low | Alternative to int8, may suit ReLU activations better |
| **Calibration data preparation** | Required for static PTQ to compute scale/zero-point | Low | Subset of training/validation data (100-512 samples typical) |
| **MinMax calibration method** | Simplest calibration method, baseline approach | Low | Uses min/max values from calibration data |
| **Quantized model accuracy evaluation** | Must measure accuracy impact of quantization | Low | Reuses existing CIFAR-10 evaluation infrastructure |
| **Per-class accuracy breakdown** | Identify which classes suffer most from quantization | Low | Already implemented for full-precision models |
| **Accuracy delta reporting** | Quantified accuracy loss vs 87.19% baseline | Low | Critical for assessing quantization viability |

## Differentiators

Features that provide deeper insight. Not expected, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Multiple calibration methods** | Compare MinMax vs Entropy vs Percentile | Medium | ONNX Runtime supports 3 methods, helps find optimal calibration |
| **Per-channel weight quantization** | Can improve accuracy for models with large weight ranges | Medium | ONNX Runtime supports this, may need reduce_range on AVX2/AVX512 |
| **Calibration set size sensitivity** | Test 100 vs 256 vs 512 samples impact on accuracy | Low | Understand minimum viable calibration data |
| **Symmetric vs asymmetric quantization** | Compare different quantization schemes | Medium | PyTorch allows configuring per-layer quantization schemes |
| **Observer comparison (PyTorch)** | MinMax vs MovingAverageMinMax observers | Medium | Different observers for weights vs activations recommended |
| **Quantization format comparison** | QDQ vs QOperator (ONNX Runtime) | Medium | QDQ more portable, QOperator potentially faster |
| **Per-layer quantization sensitivity** | Identify which layers degrade accuracy most | High | Requires instrumentation, useful for mixed-precision exploration |
| **Confusion matrix comparison** | Full-precision vs quantized confusion matrices | Low | Visualize which class confusions increase post-quantization |

## Anti-Features (Out of Scope)

Features to explicitly NOT build. Would expand scope beyond PTQ evaluation.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Quantization-aware training (QAT)** | Different milestone, requires retraining pipeline | Focus on PTQ only, defer QAT to future work |
| **Dynamic quantization** | Different quantization paradigm (weights-only) | Static quantization only per milestone scope |
| **Performance benchmarking** | Milestone focuses on accuracy, not inference speed | Document accuracy only, defer latency/throughput measurement |
| **Mixed-precision quantization** | Complex feature requiring per-layer sensitivity analysis | Uniform int8/uint8 only, defer mixed-precision |
| **Custom quantization schemes** | Beyond standard int8/uint8 formats | Use built-in quantization formats only |
| **Model architecture modification** | PTQ is post-training, no architecture changes | Use existing ResNet8 as-is |
| **TFLite quantization** | Out of framework scope (ONNX RT + PyTorch only) | Stick to declared frameworks |
| **Quantization for training** | Only evaluating inference quantization | Inference-only quantization |
| **Advanced calibration algorithms** | Beyond standard MinMax/Entropy/Percentile | Use built-in calibration methods |

## Expected Accuracy Impact

### Typical PTQ Accuracy Loss for 8-bit INT8 on CNNs

Based on research findings:

**General expectation:**
- INT8 PTQ on CNNs: **< 1% accuracy loss** on standard benchmarks (ImageNet)
- ResNet architectures are relatively robust to quantization compared to efficient models like MobileNet
- Smaller networks may experience slightly higher degradation than larger models

**For ResNet8 on CIFAR-10 (87.19% baseline):**

| Scenario | Expected Accuracy | Accuracy Loss | Confidence |
|----------|------------------|---------------|------------|
| **Best case** (optimal calibration) | 86.5-87.0% | 0.2-0.7% | MEDIUM |
| **Typical case** (MinMax, 256 samples) | 85.5-86.5% | 0.7-1.7% | MEDIUM |
| **Worst case** (poor calibration) | 83-85% | 2-4% | LOW |

**Key factors affecting accuracy:**

1. **Calibration quality**: Representative calibration data is critical
   - 100-512 samples typical, must cover data distribution
   - Random or non-representative data can cause severe degradation

2. **Calibration method**: MinMax < Entropy ≈ Percentile
   - MinMax is simplest but may be suboptimal
   - Entropy/Percentile can improve accuracy by 0.5-1%

3. **Quantization scheme**:
   - Per-channel > Per-tensor (0.5-1% improvement possible)
   - Symmetric for weights, asymmetric for activations (standard)

4. **Model size**: Smaller models (like ResNet8) may lose more than larger ResNets
   - ResNet8 is relatively small (8 layers)
   - Expect upper end of 1-2% loss range

### INT8 vs UINT8 Comparison

| Aspect | INT8 (signed) | UINT8 (unsigned) |
|--------|---------------|------------------|
| **Range** | [-128, 127] | [0, 255] |
| **Weight quantization** | Preferred (symmetric, centered at 0) | Not typical |
| **Activation quantization** | Standard (asymmetric with zero-point) | Better for post-ReLU (always positive) |
| **Hardware support** | Ubiquitous (INT8xINT8 acceleration) | Variable, some backends limited |
| **Expected accuracy** | Baseline | Similar to INT8 for activations |
| **Recommendation** | Default choice | Test as alternative for activations |

**Practical guidance:**
- Start with INT8 for both weights and activations (most compatible)
- Test UINT8 for activations if INT8 accuracy is borderline
- ResNet8 uses ReLU activations (always positive), so UINT8 may theoretically help
- Hardware support varies: verify ONNX Runtime and PyTorch backends support UINT8

## Feature Dependencies

### Existing Features (Already Implemented)

From milestone v1.0 and v1.1:
- CIFAR-10 dataset loading and preprocessing
- ONNX model loading and inference (ONNX Runtime)
- PyTorch model loading and inference (converted via onnx2torch)
- Per-class accuracy evaluation
- 87.19% full-precision baseline established

### New Feature Dependencies

```
Calibration Data Preparation
  └─> MinMax Calibration (baseline)
       ├─> ONNX Runtime Static Quantization (int8)
       │    └─> Quantized ONNX Model Accuracy Evaluation
       │         └─> Accuracy Delta Analysis
       ├─> ONNX Runtime Static Quantization (uint8)
       │    └─> Quantized ONNX Model Accuracy Evaluation
       │         └─> Accuracy Delta Analysis
       ├─> PyTorch Static Quantization (int8)
       │    └─> Quantized PyTorch Model Accuracy Evaluation
       │         └─> Accuracy Delta Analysis
       └─> PyTorch Static Quantization (uint8)
            └─> Quantized PyTorch Model Accuracy Evaluation
                 └─> Accuracy Delta Analysis

Optional (Differentiators):
  Multiple Calibration Methods
  Per-Channel Quantization
  Calibration Set Size Sensitivity
```

## MVP Recommendation

For milestone v1.2 PTQ evaluation, prioritize:

### Phase 1: ONNX Runtime PTQ (Core)
1. Calibration data preparation (256 CIFAR-10 samples)
2. MinMax calibration method
3. int8 quantization (weights + activations)
4. Quantized model accuracy evaluation
5. Accuracy delta reporting vs 87.19% baseline

### Phase 2: PyTorch PTQ (Core)
1. Reuse calibration data from Phase 1
2. PT2E export-based quantization workflow
3. int8 quantization (weights + activations)
4. Quantized model accuracy evaluation
5. Accuracy delta reporting vs 87.19% baseline

### Phase 3: uint8 Exploration (Optional)
1. ONNX Runtime uint8 quantization
2. PyTorch uint8 quantization
3. Compare int8 vs uint8 accuracy

### Defer to Post-MVP
- Multiple calibration methods (Entropy, Percentile)
- Per-channel quantization
- Calibration set size sensitivity
- Observer comparison (PyTorch)
- Quantization format comparison (QDQ vs QOperator)
- Per-layer sensitivity analysis
- Confusion matrix comparison

**Rationale:**
- Phase 1+2 provide complete PTQ evaluation for both frameworks (milestone goal)
- int8 is industry standard, most compatible format
- MinMax is simplest calibration, sufficient for initial assessment
- Phase 3 adds uint8 as bonus if time permits (ResNet8 has ReLU, may benefit)
- Differentiators deferred: nice-to-have but not critical for initial PTQ evaluation

## Implementation Notes

### ONNX Runtime Static Quantization

**API:** `onnxruntime.quantization.quantize_static()`

**Key parameters:**
- `model_input`: Path to full-precision ONNX model
- `model_output`: Path to save quantized model
- `calibration_data_reader`: Custom class providing calibration samples
- `quant_format`: QDQ (portable) vs QOperator (potentially faster)
- `activation_type`: QuantType.QUInt8 or QuantType.QInt8
- `weight_type`: QuantType.QInt8 (typical)
- `calibrate_method`: MinMax, Entropy, or Percentile

**Gotchas:**
- Zero-point must represent FP32 zero exactly (critical for zero-padding in CNNs)
- AVX2/AVX512 U8S8 format may have saturation issues (use reduce_range)
- Per-channel quantization may need reduce_range on x86-64
- Some quantized models run slower than FP32 if backend doesn't support ops

### PyTorch Static Quantization

**API:** New PT2E (PyTorch 2 Export) approach recommended

**Workflow:**
1. `torch.export.export()` - Capture model in graph mode
2. `prepare_pt2e()` - Fold BatchNorm, insert observers
3. Run calibration data through model
4. `convert_pt2e()` - Produce quantized model

**Key considerations:**
- Calibration data quality critical (100 mini-batches typical)
- Observer selection matters:
  - Weights: Symmetric per-channel + MinMax observer
  - Activations: Asymmetric per-tensor + MovingAverageMinMax observer
- BatchNorm folding into Conv2d (automatic in prepare_pt2e)
- Module naming must not overlap (causes erroneous calibration)

**Gotchas:**
- Random calibration data = bad quantization parameters (validate with real data)
- Not all modules may be calibrated if model has dynamic control flow
- Distribution drift may require re-calibration over time
- Harder to debug than FP32 models (mismatched scale/zero-point issues)

### Calibration Data Preparation

**Size:** 100-512 samples (256 recommended starting point)

**Sampling strategy:**
- Random subset of training or validation set
- Must be representative of inference distribution
- Stratified sampling (equal samples per class) recommended for CIFAR-10

**CIFAR-10 specific:**
- 10 classes, 256 samples = ~25-26 samples per class
- Use validation set (avoid test set for calibration)
- Apply same preprocessing as full-precision model (raw pixel values 0-255)

## Sources

### High Confidence (Official Documentation)
- [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [PyTorch 2 Export Post Training Quantization Tutorial](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html)
- [PyTorch Static Quantization Documentation](https://docs.pytorch.org/ao/stable/static_quantization.html)
- [PyTorch Quantization Overview](https://docs.pytorch.org/docs/stable/quantization.html)

### Medium Confidence (Verified Sources)
- [Post-training Quantization Google AI Edge](https://ai.google.dev/edge/litert/models/post_training_quantization)
- [Practical Quantization in PyTorch Blog](https://pytorch.org/blog/quantization-in-practice/)
- [Neural Network Quantization in PyTorch](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/)
- [PyTorch Static Quantization Tutorial](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

### Low Confidence (Research Papers and Forums)
- [INT8 Quantization Fundamentals](https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/quantization-fundamentals)
- [Quantization Data Types Discussion](https://apxml.com/courses/practical-llm-quantization/chapter-1-foundations-model-quantization/integer-data-types)
- [PyTorch Forums: Expected INT8 Accuracies](https://discuss.pytorch.org/t/expected-int8-accuracies-on-imagenet-1k-resnet-qat/187227)
- [GitHub: Static Quantization Calibration Issues](https://github.com/pytorch/pytorch/issues/45185)

---

**Overall Confidence:** MEDIUM

- **HIGH** confidence on API workflows and official quantization methods (official docs verified)
- **MEDIUM** confidence on expected accuracy impact (multiple sources agree on <1% for INT8, but ResNet8 is smaller than benchmarked models)
- **LOW** confidence on specific ResNet8/CIFAR-10 accuracy expectations (extrapolated from larger models)

**Validation needed:**
- Actual ResNet8 quantization accuracy (empirical testing required)
- UINT8 hardware support in current ONNX Runtime/PyTorch versions
- Optimal calibration method for this specific model (MinMax vs Entropy vs Percentile)
