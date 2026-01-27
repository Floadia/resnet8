# ResNet8 Model Evaluation

## What This Is

Multi-framework evaluation of ResNet8 for CIFAR-10, converted from the MLCommons TinyMLPerf Keras implementation. Supports ONNX Runtime and PyTorch inference with accuracy validation on CIFAR-10 test set.

## Core Value

Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10).

## Current Milestone: v1.1 PyTorch Conversion

**Goal:** Convert existing ONNX model to PyTorch and validate accuracy on CIFAR-10

**Target features:**
- ONNX → PyTorch conversion using onnx2torch
- PyTorch model evaluation on CIFAR-10
- Accuracy >85% (same threshold as ONNX)

## Requirements

### Validated

- ✓ Keras .h5 → ONNX conversion using tf2onnx — v1.0
- ✓ ONNX model verification (structure, shapes) — v1.0
- ✓ CIFAR-10 evaluation with ONNX Runtime (87.19%) — v1.0
- ✓ Per-class accuracy breakdown — v1.0
- ✓ Accuracy >85% on CIFAR-10 test set — v1.0

### Active

- [ ] ONNX → PyTorch conversion using onnx2torch
- [ ] PyTorch model verification
- [ ] CIFAR-10 evaluation with PyTorch
- [ ] Per-class accuracy breakdown (PyTorch)
- [ ] Accuracy >85% on CIFAR-10 test set (PyTorch)

### Out of Scope

- Training from scratch — using pretrained weights only
- TFLite/quantized model support — only full-precision evaluation
- Performance benchmarking — accuracy only, not inference speed
- Custom datasets — CIFAR-10 only
- Manual weight transfer — using onnx2torch for conversion

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

- **Frameworks**: ONNX Runtime, PyTorch (targets), TensorFlow/Keras (source)
- **Conversion tools**: tf2onnx for Keras→ONNX, onnx2torch for ONNX→PyTorch
- **Dataset**: CIFAR-10 (included in reference project)
- **Accuracy target**: >85% to validate successful conversion

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ONNX as intermediate | Standard interchange format, well-supported | ✓ Good |
| tf2onnx for Keras→ONNX | Standard tool, well-maintained | ✓ Good |
| Separate converter/eval scripts | Reusability and clarity | ✓ Good |
| Raw pixel values (0-255) | Match Keras training preprocessing | ✓ Good |
| onnx2torch for ONNX→PyTorch | Leverage existing ONNX model | — Pending |

---
*Last updated: 2026-01-27 after v1.1 milestone start*
