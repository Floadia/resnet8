# ResNet8 PyTorch Evaluation

## What This Is

A PyTorch implementation of ResNet8 for CIFAR-10 evaluation, ported from the MLCommons TinyMLPerf Keras implementation. Converts pretrained Keras weights to PyTorch and validates accuracy on CIFAR-10 test set.

## Core Value

Accurate weight conversion — PyTorch model must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10).

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] PyTorch ResNet8 model matching Keras architecture exactly
- [ ] Weight converter: Keras .h5 → PyTorch .pt
- [ ] CIFAR-10 evaluation script with accuracy reporting
- [ ] Accuracy >85% on CIFAR-10 test set

### Out of Scope

- Training from scratch — using pretrained weights only
- TFLite/quantized model support — only full-precision evaluation
- Performance benchmarking — accuracy only, not inference speed
- Custom datasets — CIFAR-10 only

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

- **Framework**: PyTorch (target), TensorFlow/Keras (source for weights)
- **Dataset**: CIFAR-10 (included in reference project)
- **Accuracy target**: >85% to validate successful conversion

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate model/converter/eval scripts | Reusability and clarity | — Pending |
| Match Keras layer ordering exactly | Weight conversion requires 1:1 mapping | — Pending |

---
*Last updated: 2025-01-27 after initialization*
