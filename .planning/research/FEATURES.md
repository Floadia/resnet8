# Feature Landscape: Keras→PyTorch Model Conversion

**Domain:** Deep learning model conversion (Keras .h5 → PyTorch .pt)
**Researched:** 2026-01-27
**Confidence:** MEDIUM (WebSearch verified with community sources, some LOW confidence items flagged)

## Table Stakes

Features users expect. Missing = conversion fails or produces incorrect results.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Architecture reconstruction** | Must replicate Keras model structure in PyTorch | Medium | Manual layer-by-layer mapping required; no reliable automated tools exist |
| **Weight extraction from .h5** | Source of trained parameters | Low | Keras provides `model.load_weights()` API; straightforward file access |
| **Weight shape transformation** | Keras and PyTorch use different tensor layouts | High | Conv2D: Keras (H,W,In,Out) → PyTorch (Out,In,H,W); Dense: Keras (Out,In) → PyTorch (In,Out) |
| **BatchNorm parameter mapping** | Different ordering and naming conventions | Medium | Keras [gamma, beta, mean, var] → PyTorch [weight, bias, running_mean, running_var] |
| **Layer name mapping** | Must correspond layers between frameworks | Medium | Manual mapping dictionary; naming conventions differ between frameworks |
| **Numerical validation** | Verify conversion produces correct outputs | Medium | Compare predictions on same inputs; critical for detecting silent errors |
| **Accuracy validation** | Confirm model performance post-conversion | Low | Run evaluation on test set; >85% accuracy threshold for this project |

## Differentiators

Features that improve conversion quality/reliability but aren't strictly required.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **ONNX intermediate format** | Framework-agnostic conversion path | Medium | Keras→ONNX→PyTorch reduces direct mapping complexity; tools: tf2onnx, onnx2pytorch |
| **Per-layer output comparison** | Detect exactly where conversion diverges | Medium | Forward pass both models, compare intermediate activations layer-by-layer |
| **Automated test suite** | Catch regressions during conversion iterations | Low | Property tests: shape matching, weight count, output consistency |
| **Weight freeze verification** | Ensure no accidental training during validation | Low | Check `requires_grad=False` or use `model.eval()` mode |
| **Epsilon consistency checking** | Prevent accumulative errors in normalization | Medium | Verify BatchNorm/LayerNorm epsilon values match between frameworks |
| **Conversion report generation** | Document what was converted and validation results | Low | Markdown/JSON report with layer mapping, accuracy metrics, warnings |
| **Side-by-side inference comparison** | Visual/quantitative output comparison | Low | Run same images through both models, diff predictions |
| **State dict serialization** | Save converted PyTorch weights | Low | `torch.save(model.state_dict(), 'model.pt')` for reusability |

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Fully automated conversion tools** | No reliable tools exist (2026); nn-transfer is archived/broken with latest versions | Manual architecture + weight mapping with validation |
| **Training from scratch instead of converting** | Defeats purpose; requires labeled data, compute, hyperparameter tuning | Convert pretrained weights; only train if conversion fails validation |
| **Channel order auto-detection** | Error-prone; NCHW vs NHWC differences cause silent failures | Explicitly document and handle (H,W,C) → (C,H,W) transformations |
| **Skipping numerical validation** | Conversion bugs manifest as accuracy degradation (98%→60% reported) | Always validate outputs match on same inputs within tolerance |
| **Global epsilon defaults** | Different frameworks use different defaults (1e-5 vs 1e-3); accumulates error | Explicitly set epsilon to match source model |
| **Assuming layer ordering is consistent** | Keras uses different initialization and layer chaining than PyTorch | Manually verify each layer maps correctly; use named modules |
| **Converting without understanding architecture** | Residual connections, shortcuts need manual attention | Study Keras model structure first; document residual paths |
| **Using outdated conversion libraries** | pytorch2keras (2022), nn-transfer (archived) produce incorrect outputs | Use 2026-current manual conversion practices |

## Feature Dependencies

```
Weight Extraction (.h5 file)
    ↓
Architecture Reconstruction (PyTorch model definition)
    ↓
Layer Name Mapping (Keras layers → PyTorch modules)
    ↓
Weight Shape Transformation (tensor layout conversion)
    ↓
BatchNorm Parameter Mapping (specific to normalization layers)
    ↓
State Dict Loading (assign weights to PyTorch model)
    ↓
Numerical Validation (same inputs → same outputs?)
    ↓
Accuracy Validation (test set performance)
```

**Critical path:** Architecture must be recreated before weights can be loaded. Weight shapes must be transformed before assignment. Validation must happen after loading to catch errors.

**Optional path (ONNX approach):**
```
Keras .h5 Model
    ↓ (tf2onnx)
ONNX Intermediate Format
    ↓ (onnx2pytorch)
PyTorch Model
    ↓
Validation (same as manual path)
```

## MVP Recommendation

For MVP (ResNet8 CIFAR-10 conversion), prioritize:

1. **Architecture reconstruction** - Match Keras model exactly (Conv2D, BatchNorm, residual connections)
2. **Weight shape transformation** - Handle Conv2D and Dense layer tensor layout differences
3. **BatchNorm parameter mapping** - Critical for ResNet8 (uses BN after every conv)
4. **Numerical validation** - Compare outputs on 10-100 test images (tolerance: 1e-5)
5. **Accuracy validation** - Run full CIFAR-10 test set; target >85%

Defer to post-MVP (if needed):

- **ONNX conversion path**: Investigate only if manual conversion fails validation (reason: adds complexity, extra dependencies)
- **Per-layer output comparison**: Use only for debugging if accuracy validation fails (reason: time-consuming, not needed if end-to-end works)
- **Automated test suite**: Add after successful first conversion (reason: premature for one-time conversion)
- **Conversion report generation**: Manual notes sufficient for single model (reason: overhead not justified)

## Domain-Specific Notes

### ResNet8 Conversion Considerations

This project converts a **ResNet8 with residual connections**. Key features specific to this architecture:

1. **Residual/shortcut connections:** Must preserve exact layer connections (identity vs 1×1 conv shortcuts)
2. **Strided convolutions in shortcuts:** Stack 2/3 use stride=2 convolutions for downsampling; shortcut must match
3. **BatchNorm placement:** Keras places BN after Conv2D but before activation; PyTorch convention varies - must match exactly
4. **L2 regularization:** Keras applies during training; not stored in weights, irrelevant for inference-only conversion
5. **He normal initialization:** Used in training; not relevant for pretrained weight conversion

### Validation Strategy for This Project

Given the 85% accuracy target:

- **Minimum viable validation:** Run converted model on CIFAR-10 test set (10k images); report accuracy
- **If accuracy < 85%:** Implement per-layer output comparison to identify divergence point
- **If accuracy ≈ 85%:** Conversion successful; no deeper validation needed

### Known Failure Modes

Based on community reports (LOW confidence - unverified):

1. **Padding mismatch:** Keras `padding='same'` vs PyTorch integer padding causes shape errors (98%→60% accuracy drop reported)
2. **Stride inconsistencies:** Mismatched strides in Conv2D cause resolution mismatches
3. **Weights loaded from wrong model:** Source model not actually loaded (e.g., untrained weights used)
4. **BatchNorm epsilon differences:** Default epsilon differs; causes accumulative error over deep networks
5. **Gradient state bleeding:** `model.train()` vs `model.eval()` mode affects BatchNorm; must use eval for inference

## Sources

**Conversion Tools & Methods:**
- [nn-transfer: Convert trained PyTorch models to Keras (archived)](https://github.com/gzuidhof/nn-transfer) - MEDIUM confidence
- [deep-learning-model-convertor: Multi-framework converter](https://github.com/ysh329/deep-learning-model-convertor) - MEDIUM confidence
- [How to Transfer a Simple Keras Model to PyTorch - The Hard Way](https://gereshes.com/2019/06/24/how-to-transfer-a-simple-keras-model-to-pytorch-the-hard-way/) - MEDIUM confidence

**Weight Mapping & Tensor Layouts:**
- [Copying weight tensors from PyTorch to Tensorflow (and back)](https://www.adrian.idv.hk/2022-05-21-torch2tf/) - MEDIUM confidence
- [Transferring weights from Keras to PyTorch - PyTorch Forums](https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889) - MEDIUM confidence
- [Load Keras Weight to PyTorch - Medium](https://medium.com/analytics-vidhya/load-keras-weight-to-pytorch-and-transform-keras-architecture-to-pytorch-easily-8ff5dd18b86b) - MEDIUM confidence

**BatchNorm & Normalization:**
- [Pitfalls encountered porting models to Keras from PyTorch/TensorFlow/MXNet](https://shaoanlu.wordpress.com/2019/05/23/pitfalls-encountered-porting-models-to-keras-from-pytorch-and-tensorflow/) - MEDIUM confidence
- [Different results for batchnorm with pytorch and tensorflow/keras](https://discuss.pytorch.org/t/different-results-for-batchnorm-with-pytorch-and-tensorflow-keras/151691) - MEDIUM confidence
- [Porting a pretrained ResNet from Pytorch to Tensorflow 2.0](https://dmolony3.github.io/Pytorch-to-Tensorflow.html) - MEDIUM confidence

**ONNX Conversion:**
- [The Complete Guide to Converting Machine Learning Models to ONNX Format in 2025](https://medium.com/@liutaurasog/the-complete-guide-to-converting-machine-learning-models-to-onnx-format-in-2025-54104cc4aa85) - MEDIUM confidence
- [onnx2tf: Convert ONNX to TensorFlow/Keras](https://github.com/PINTO0309/onnx2tf) - MEDIUM confidence
- [tensorflow-onnx: Convert TensorFlow/Keras to ONNX](https://github.com/onnx/tensorflow-onnx) - MEDIUM confidence
- [Converting PyTorch Models to Keras via ONNX](https://www.codegenes.net/blog/onnx-pytorch-to-keras/) - MEDIUM confidence

**Validation & Testing:**
- [Reproducibly benchmarking Keras and PyTorch models](https://github.com/cgnorthcutt/benchmarking-keras-pytorch) - MEDIUM confidence
- [Towards Reproducibility: Benchmarking Keras and PyTorch](https://l7.curtisnorthcutt.com/towards-reproducibility-benchmarking-keras-pytorch) - MEDIUM confidence
- [Performance comparison of medical image classification systems using TensorFlow Keras, PyTorch, and JAX (2025)](https://arxiv.org/html/2507.14587v1) - MEDIUM confidence

**Common Pitfalls:**
- [TF-Keras to PyTorch Model conversion target and input size mismatch](https://discuss.pytorch.org/t/tf-keras-to-pytorch-model-conversion-target-and-input-size-mismatch/142379) - LOW confidence
- [KERAS TO pytorch model conversion - PyTorch Forums](https://discuss.pytorch.org/t/keras-to-pytorch-model-conversion/155153) - LOW confidence
- [Converting TensorFlow Model Weights to PyTorch Weights](https://www.codegenes.net/blog/how-to-convert-tensorflow-model-weights-to-pytorch-weights/) - MEDIUM confidence

---

**Confidence Assessment:**

- **Table Stakes:** MEDIUM - Based on multiple community sources and documented conversion workflows; core requirements verified across sources
- **Differentiators:** MEDIUM - Features mentioned in multiple conversion guides; ONNX approach verified via official repos
- **Anti-Features:** MEDIUM-LOW - Based on community reports and archived project states; specific accuracy numbers (98%→60%) are LOW confidence
- **Dependencies:** HIGH - Logical ordering derived from conversion process; verified against multiple sources

**Gaps:**
- No official PyTorch or Keras documentation specifically about cross-framework conversion (both assume native workflows)
- Most sources are 2019-2024; no significant 2026-specific changes found
- Specific epsilon values and accuracy degradation numbers are anecdotal (LOW confidence)
- ONNX approach not tested for this specific ResNet8 architecture
