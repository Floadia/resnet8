/**
 * ResNet8 CIFAR-10 Model Conversion and Quantization
 *
 * Multi-framework model conversion from Keras to ONNX and PyTorch, with Post-Training
 * Quantization (PTQ) support for model compression.
 *
 * ## Project Purpose
 *
 * **Source:** MLCommons TinyMLPerf ResNet8 model trained on CIFAR-10
 *
 * **Goal:** Enable cross-framework inference with high accuracy retention (>85%)
 *
 * **Quantization:** Static PTQ with int8/uint8 support for reduced model size while
 * maintaining accuracy
 *
 * ## Quick Results
 *
 * | Model | Accuracy | Size | Reduction |
 * |-------|----------|------|-----------|
 * | FP32 baseline (ONNX) | 87.19% | 315KB | - |
 * | ONNX Runtime uint8 | 86.75% | 123KB | 61% |
 * | ONNX Runtime int8 | 85.58% | 123KB | 61% |
 * | PyTorch int8 | 85.68% | 165KB | 52% |
 *
 * **Recommendation:** ONNX Runtime uint8 provides best accuracy-to-size ratio
 * (-0.44% accuracy drop, 61% size reduction)
 *
 * ## Architecture
 *
 * **ResNet8 for CIFAR-10:**
 * - **Input:** 32×32×3 RGB images
 * - **Architecture:** 3 residual stacks (16→32→64 filters)
 *   - Initial: Conv2D(16, 3×3) → BatchNorm → ReLU
 *   - Stack 1: 2× residual blocks (16 filters, identity shortcut)
 *   - Stack 2: 2× residual blocks (32 filters, stride=2, 1×1 conv shortcut)
 *   - Stack 3: 2× residual blocks (64 filters, stride=2, 1×1 conv shortcut)
 *   - Head: Global Average Pooling → Dense(10, softmax)
 * - **Output:** 10-class probability distribution
 *
 * ## Modules
 *
 * This documentation covers two main modules:
 *
 * ### {@link scripts}
 * Command-line tools for model conversion, evaluation, and quantization:
 * - Keras → ONNX conversion
 * - ONNX → PyTorch conversion
 * - Model evaluation on CIFAR-10
 * - Static quantization (int8/uint8)
 * - Operation extraction and validation
 * - Graph visualization
 *
 * ### {@link playground}
 * Interactive utilities for Marimo notebooks:
 * - Cached model loading (prevents memory leaks)
 * - Layer name inspection
 * - Model variant management
 *
 * @module
 */

// Re-export all script utilities
export * from "./scripts.ts";

// Re-export all playground utilities
export * from "./playground.ts";
