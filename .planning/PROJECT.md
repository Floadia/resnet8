# ResNet8 Model Evaluation

## What This Is

Multi-framework evaluation of ResNet8 for CIFAR-10, converted from the MLCommons TinyMLPerf Keras implementation. Supports ONNX Runtime and PyTorch inference with accuracy validation on CIFAR-10 test set.

## Core Value

Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10).

## Current Milestone: v1.5 Intermediate Value Visualizer

**Goal:** Add intermediate activation capture and visualization to weight_visualizer.py

**Target features:**
- Run inference with a CIFAR-10 sample or random input through PyTorch models
- Capture intermediate activations at each layer using forward hooks
- Visualize activation distributions (histograms) and statistics per layer
- Compare activations across layers

## Requirements

### Validated

- ✓ Keras .h5 → ONNX conversion using tf2onnx — v1.0
- ✓ ONNX model verification (structure, shapes) — v1.0
- ✓ CIFAR-10 evaluation with ONNX Runtime (87.19%) — v1.0
- ✓ Per-class accuracy breakdown — v1.0
- ✓ Accuracy >85% on CIFAR-10 test set — v1.0
- ✓ ONNX → PyTorch conversion using onnx2torch — v1.1
- ✓ PyTorch model verification — v1.1
- ✓ CIFAR-10 evaluation with PyTorch (87.19%) — v1.1
- ✓ Per-class accuracy breakdown (PyTorch) — v1.1
- ✓ Accuracy >85% on CIFAR-10 test set (PyTorch) — v1.1
- ✓ ONNX Runtime static quantization (int8/uint8) — v1.2
- ✓ PyTorch static quantization (int8) — v1.2
- ✓ Calibration data preparation (CIFAR-10 subset) — v1.2
- ✓ Quantized model accuracy evaluation — v1.2
- ✓ Accuracy delta analysis (quantized vs 87.19% baseline) — v1.2

### Active

- [ ] Intermediate activation capture using PyTorch forward hooks in weight_visualizer.py
- [ ] Input selection (CIFAR-10 sample or random input) for inference
- [ ] Activation distribution visualization (histograms, statistics) per layer
- [ ] Integration with existing weight/bias visualization

### Out of Scope

- Training from scratch — using pretrained weights only
- Quantization-aware training (QAT) — PTQ only
- Performance benchmarking — accuracy only, not inference speed
- Custom datasets — CIFAR-10 only
- Dynamic quantization — static quantization only
- TFLite quantization — ONNX Runtime and PyTorch only

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
*Last updated: 2026-02-16 after v1.5 milestone start*
