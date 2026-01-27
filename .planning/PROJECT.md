# ResNet8 ONNX Evaluation

## What This Is

ONNX-based evaluation of ResNet8 for CIFAR-10, converted from the MLCommons TinyMLPerf Keras implementation. Converts pretrained Keras model to ONNX and validates accuracy on CIFAR-10 test set using ONNX Runtime.

## Core Value

Accurate Keras→ONNX conversion — ONNX model must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10).

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Keras .h5 → ONNX conversion using tf2onnx
- [ ] ONNX model verification (structure, shapes)
- [ ] CIFAR-10 evaluation with ONNX Runtime
- [ ] Per-class accuracy breakdown
- [ ] Accuracy >85% on CIFAR-10 test set

### Out of Scope

- Training from scratch — using pretrained weights only
- TFLite/quantized model support — only full-precision evaluation
- Performance benchmarking — accuracy only, not inference speed
- Custom datasets — CIFAR-10 only
- PyTorch conversion — ONNX only for v1 (PyTorch deferred to v2)

## Context

**Source project:** MLCommons TinyMLPerf image classification benchmark
- Location: `/mnt/ext1/references/tiny/benchmark/training/image_classification`
- Trained model: `trained_models/pretrainedResnet.h5`

**Architecture (from keras_model.py):**
- Input: 32×32×3 (CIFAR-10 images)
- Initial: Conv2D(16, 3×3) → BN → ReLU
- Stack 1: 2× Conv2D(16, 3×3) + identity shortcut
- Stack 2: 2× Conv2D(32, 3×3, stride=2) + 1×1 conv shortcut
- Stack 3: 2× Conv2D(64, 3×3, stride=2) + 1×1 conv shortcut
- Head: AvgPool → Dense(10, softmax)

**Key details:**
- Uses L2 regularization (1e-4) on conv kernels
- BatchNorm after each conv (before ReLU in residual path)
- He normal initialization
- Shortcut convs use 1×1 kernel with stride 2 for dimension matching

## Constraints

- **Framework**: ONNX Runtime (target), TensorFlow/Keras (source)
- **Conversion tool**: tf2onnx for Keras→ONNX
- **Dataset**: CIFAR-10 (included in reference project)
- **Accuracy target**: >85% to validate successful conversion

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ONNX instead of PyTorch | User plans future ONNX evaluation work | — Pending |
| tf2onnx for conversion | Standard tool for Keras→ONNX, well-maintained | — Pending |
| Separate converter/eval scripts | Reusability and clarity | — Pending |

---
*Last updated: 2025-01-27 after initialization*
