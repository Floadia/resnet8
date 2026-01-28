# ResNet8 CIFAR-10 Post-Training Quantization Analysis

## Executive Summary

This document presents the quantization analysis for the ResNet8 model trained on CIFAR-10, comparing Post-Training Quantization (PTQ) results across ONNX Runtime and PyTorch frameworks.

**Key Finding:** ONNX Runtime uint8 achieves the best accuracy retention at 86.75% (-0.44% from baseline), while all quantized models meet the >85% accuracy threshold.

**Deployment Recommendation:** Use ONNX Runtime uint8 for best accuracy-to-size ratio. For PyTorch-only deployments, int8 quantization is acceptable with minimal accuracy loss (-1.51%).

## Comparison Table

| Framework | Data Type | Accuracy | Delta vs Baseline | Model Size | Size Reduction |
|-----------|-----------|----------|-------------------|------------|----------------|
| ONNX (FP32 Baseline) | float32 | 87.19% | - | 315KB | - |
| PyTorch (FP32 Baseline) | float32 | 87.19% | - | 345KB | - |
| ONNX Runtime | int8 | 85.58% | -1.61% | 123KB | 61% |
| ONNX Runtime | uint8 | 86.75% | -0.44% | 123KB | 61% |
| PyTorch | int8 | 85.68% | -1.51% | 165KB | 52% |
| PyTorch | uint8 | N/A | N/A | N/A | Not supported (fbgemm requires qint8 weights) |

## Accuracy Analysis

### Accuracy Drop Assessment

**No configurations exceed the 5% accuracy drop threshold.**

All quantized models maintain accuracy within acceptable ranges:

| Configuration | Accuracy Drop | Status |
|---------------|---------------|--------|
| ONNX Runtime uint8 | -0.44% | Pass |
| PyTorch int8 | -1.51% | Pass |
| ONNX Runtime int8 | -1.61% | Pass |

### Per-Class Impact (Int8 vs Uint8)

Analysis of per-class accuracy revealed:
- Most classes maintain within 3% of baseline accuracy
- The "dog" class shows the largest variance:
  - Int8 (ONNX): 74.2% -> 68.0% (-6.2% drop)
  - Uint8 (ONNX): 74.2% -> 74.4% (+0.2% improvement)
- Uint8 quantization provides more uniform accuracy across classes

## Size Comparison

### Original Model Sizes
- **ONNX FP32:** 315KB (reference for ONNX quantization)
- **PyTorch FP32:** 345KB (reference for PyTorch quantization)

### Quantized Model Sizes

| Model | Original Size | Quantized Size | Reduction |
|-------|---------------|----------------|-----------|
| ONNX int8 | 315KB | 123KB | 61% |
| ONNX uint8 | 315KB | 123KB | 61% |
| PyTorch int8 | 345KB | 165KB | 52% |

**Note:** PyTorch int8 model is larger than ONNX equivalents due to TorchScript format overhead. ONNX uses QDQ (QuantizeLinear/DequantizeLinear) format which is more compact.

## Framework Comparison

### ONNX Runtime

- **Supported types:** Both int8 and uint8
- **Better uint8 accuracy:** 86.75% vs 85.58% (int8)
- **Format:** QDQ (QuantizeLinear/DequantizeLinear operators)
- **Calibration:** MinMax method with per-tensor quantization

### PyTorch

- **Supported types:** int8 only (fbgemm backend limitation)
- **Quantization mode:** FX graph mode (required for onnx2torch models)
- **Serialization:** TorchScript (JIT tracing) for model portability
- **fbgemm limitation:** Quantized convolutions require qint8 (signed 8-bit) weights; uint8-only quantization is not supported

### Calibration Details

Both frameworks used the same calibration dataset:
- **Samples:** 1000 stratified samples from CIFAR-10 training set
- **Distribution:** 100 samples per class (balanced)
- **Preprocessing:** Float32, 0-255 range, NHWC format (matching training)
- **Method:** Static post-training quantization

## Recommendation

### Best Overall: ONNX Runtime uint8
- **Accuracy:** 86.75% (-0.44% from baseline)
- **Size:** 123KB (61% reduction)
- **Rationale:** Best accuracy retention with maximum size reduction

### PyTorch Deployment: int8
- **Accuracy:** 85.68% (-1.51% from baseline)
- **Size:** 165KB (52% reduction)
- **Rationale:** Only supported option; acceptable accuracy loss

### Summary

| Use Case | Recommended Model | Accuracy | Size |
|----------|-------------------|----------|------|
| Best accuracy | ONNX Runtime uint8 | 86.75% | 123KB |
| Best compression | ONNX Runtime int8/uint8 | 85.58%/86.75% | 123KB |
| PyTorch ecosystem | PyTorch int8 | 85.68% | 165KB |

**All quantized models meet the project requirements:**
- Accuracy > 85% threshold
- Accuracy drop < 5% from baseline

## Methodology Notes

### Baseline
- Model: ResNet8 trained on CIFAR-10
- Accuracy: 87.19% on test set (10,000 images)
- Source: Original Keras model converted via tf2onnx and onnx2torch

### Quantization Type
- **Method:** Static post-training quantization (PTQ)
- **Calibration:** Offline calibration with training data samples
- **No fine-tuning:** Pure post-training approach

### Calibration Data
- **Source:** CIFAR-10 training set
- **Samples:** 1000 (100 per class, stratified)
- **Purpose:** Collect activation statistics for scale/zero-point calculation
- **Data separation:** Calibration uses training data, evaluation uses test data

### Tools and Versions
- **ONNX Runtime:** onnxruntime.quantization with quantize_static()
- **PyTorch:** torch.ao.quantization.quantize_fx with FX graph mode
- **Calibration method:** MinMax (ONNX), default observer (PyTorch)

---

*Generated as part of v1.2 PTQ Evaluation milestone*
*Document: docs/QUANTIZATION_ANALYSIS.md*
*Date: 2026-01-28*
