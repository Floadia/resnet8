# Technology Stack: Keras→PyTorch Model Conversion

**Project:** ResNet8 CIFAR-10 Model Conversion
**Researched:** 2026-01-27
**Confidence:** HIGH

## Executive Summary

For converting a pretrained Keras ResNet8 model (.h5) to PyTorch and evaluating on CIFAR-10, the recommended approach is **manual weight conversion** with h5py for this simple architecture. ONNX-based conversion is the fallback. PyTorch 2.10.0 with torchvision 0.25 provides the evaluation infrastructure.

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **PyTorch** | 2.10.0 | Deep learning framework | Latest stable (released 2025), required for target model |
| **torchvision** | 0.25 | Vision utilities & datasets | Official PyTorch vision library, includes CIFAR-10 dataset loader |
| **Python** | >=3.10 | Runtime | Required by PyTorch 2.10.0, h5py 3.15.1, and modern ecosystem |

**Confidence:** HIGH - Verified from official PyTorch installation page (2025)

### Model Loading & Conversion
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **h5py** | 3.15.1 | Load Keras .h5 files | Latest stable (Oct 2025), reads HDF5 weight files directly |
| **tensorflow** | >=2.16 | Load Keras models (optional) | Only if using tf.keras.models.load_model() for architecture inspection |
| **numpy** | Latest compatible | Array operations | PyTorch 2.10 supports both NumPy 1.x and 2.x when built with NumPy 2.0 ABI |

**Confidence:** HIGH (h5py, numpy) / MEDIUM (tensorflow - optional dependency)

**Conversion Strategy:**
1. **Primary approach: Manual weight mapping** - For ResNet8 (small model), manually map weights from .h5 to PyTorch state_dict
2. **Fallback: ONNX pipeline** - Use tf2onnx → onnx2torch if manual conversion proves difficult

### ONNX Conversion Pipeline (Fallback)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **tf2onnx** | 1.16.1 | Keras→ONNX conversion | Latest stable (Jan 2024), official ONNX converter for TensorFlow/Keras |
| **onnx** | 1.20.1 | ONNX format | Latest stable (Jan 2026), industry standard for model exchange |
| **onnx2torch** | 1.5.15 | ONNX→PyTorch conversion | Latest stable (Aug 2024), actively maintained (ENOT-AutoDL) |

**Confidence:** HIGH (all versions verified)

**Why NOT onnx2pytorch:** Inactive maintenance (last release Nov 2021), many unresolved issues, classified as "Inactive" on PyPI health analysis.

### Evaluation & Testing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **torchmetrics** | 1.8.2 | Accuracy metrics | Latest stable, 100+ PyTorch metrics implementations |
| **pytest** | Latest | Unit testing | Simpler than unittest, better fixtures, used in PyTorch CI |

**Confidence:** HIGH

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **tqdm** | Latest | Progress bars | Dataset loading, training visualization |
| **matplotlib** | Latest | Visualization | Plot training curves, show sample predictions |

**Confidence:** MEDIUM - Common utilities but not strictly required

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Model Conversion | Manual (h5py) + onnx2torch fallback | Keras3 with PyTorch backend | Keras3 requires migrating TF model to Keras3 first, adds complexity |
| ONNX→PyTorch | onnx2torch | onnx2pytorch | Inactive (last release 2021), many unresolved issues |
| Testing | pytest | unittest | More verbose, less flexible fixtures, unittest is PyTorch-compatible but pytest preferred |
| Metrics | torchmetrics | Manual accuracy calculation | torchmetrics provides standardized, tested implementations |

## Installation

```bash
# Core PyTorch (CPU version for development)
pip install torch==2.10.0 torchvision==0.25 --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 12.6 example - adjust based on your CUDA version)
# pip install torch==2.10.0 torchvision==0.25 --index-url https://download.pytorch.org/whl/cu126

# Model loading
pip install h5py==3.15.1

# ONNX conversion pipeline (fallback approach)
pip install onnx==1.20.1 tf2onnx==1.16.1 onnx2torch==1.5.15

# TensorFlow/Keras (only if inspecting original model architecture)
# pip install tensorflow>=2.16

# Evaluation and utilities
pip install torchmetrics==1.8.2 pytest tqdm matplotlib numpy
```

### Minimal Installation (Manual Conversion Only)
```bash
# If doing manual weight conversion without ONNX
pip install torch==2.10.0 torchvision==0.25 h5py==3.15.1 torchmetrics==1.8.2 numpy
```

## Version Compatibility Notes

### NumPy Compatibility
- PyTorch 2.10.0 supports both NumPy 1.x and 2.x when built with NumPy 2.0 ABI
- Pre-built PyTorch binaries from pytorch.org should handle both versions
- If issues arise, NumPy <2.0 is safe fallback: `pip install "numpy<2"`

### Python Version
- Python >=3.10 required by both PyTorch 2.10 and h5py 3.15.1
- Recommended: Python 3.10, 3.11, or 3.12 (widespread support)
- Python 3.13+ experimental in some libraries

### CUDA Considerations
- CPU version sufficient for CIFAR-10 evaluation (10,000 32x32 images)
- For GPU: Match PyTorch CUDA version to system CUDA installation
- Available CUDA versions: 12.6, 12.8, 13.0

## Conversion Approach Recommendation

### For This Project: Manual Conversion Preferred

**Rationale:**
1. **Small model**: ResNet8 has simple architecture (8 layers, standard conv/bn/relu blocks)
2. **No exotic layers**: MLCommons TinyMLPerf ResNet8 uses standard Conv2D, BatchNorm, ReLU, Dense
3. **Transparency**: Manual conversion ensures exact understanding of weight mapping
4. **No ONNX overhead**: Avoid intermediate format and potential operator support issues

**Manual Conversion Steps:**
1. Load .h5 file with h5py: `h5_file = h5py.File('model.h5', 'r')`
2. Define equivalent PyTorch ResNet8 architecture
3. Map layer names: Keras "conv2d_1/kernel:0" → PyTorch "conv1.weight"
4. Handle format differences:
   - Keras Conv2D weights: (out, in, H, W) → PyTorch Conv2d: (out, in, H, W) [same]
   - Keras Dense weights: (in, out) → PyTorch Linear: (out, in) [transpose needed]
   - BatchNorm parameters: map beta→bias, gamma→weight, moving_mean→running_mean, moving_variance→running_var
5. Load into PyTorch: `model.load_state_dict(state_dict)`

### ONNX Fallback

Use ONNX pipeline if:
- Manual conversion hits layer incompatibilities
- Time constraint requires faster solution
- Model has custom layers not documented

**ONNX Pipeline:**
```python
# 1. Keras → ONNX
import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(keras_model)

# 2. ONNX → PyTorch
from onnx2torch import convert
pytorch_model = convert(onnx_model)
```

**Known Limitation**: onnx2torch "covers only a limited number of PyTorch/ONNX models and operations" - may require custom operator registration for edge cases.

## Quality Gates

### Conversion Success Criteria
1. Model loads without errors
2. Forward pass produces output of correct shape (batch_size, 10)
3. Output is probability distribution (softmax applied)
4. Numerical consistency: same input produces similar output (±0.01 tolerance acceptable due to framework differences)

### Evaluation Success Criteria
1. Achieves ≥85% accuracy on CIFAR-10 test set (MLCommons quality target)
2. Evaluation runs in <10 minutes on CPU (10,000 images)
3. All 10 classes evaluated correctly

## Sources

### PyTorch & torchvision
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) - Latest stable: 2.10.0 (HIGH confidence)
- [torchvision Documentation](https://docs.pytorch.org/vision/stable/index.html) - Version 0.25 (HIGH confidence)
- [CIFAR10 Dataset Documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) (HIGH confidence)

### Conversion Libraries
- [h5py PyPI](https://pypi.org/project/h5py/) - Version 3.15.1, Oct 2025 (HIGH confidence)
- [ONNX GitHub Releases](https://github.com/onnx/onnx) - Version 1.20.1, Jan 2026 (HIGH confidence)
- [tf2onnx PyPI](https://pypi.org/project/tf2onnx/) - Version 1.16.1, Jan 2024 (HIGH confidence)
- [onnx2torch PyPI](https://pypi.org/project/onnx2torch/) - Version 1.5.15, Aug 2024 (HIGH confidence)
- [Load Keras Weight to PyTorch (Medium)](https://medium.com/analytics-vidhya/load-keras-weight-to-pytorch-and-transform-keras-architecture-to-pytorch-easily-8ff5dd18b86b) - Manual conversion approach (MEDIUM confidence)

### Conversion Strategy Research
- [On the Challenge of Converting TensorFlow Models to PyTorch (Medium, Nov 2025)](https://chaimrand.medium.com/on-the-challenge-of-converting-tensorflow-models-to-pytorch-bd43a7704c62) - Discusses Keras3 and ONNX approaches (MEDIUM confidence)
- [PyTorch Forums: Keras to PyTorch Conversion](https://discuss.pytorch.org/t/keras-to-pytorch-model-conversion/155153) - Community discussion (LOW confidence)

### MLCommons TinyMLPerf
- [MLCommons Tiny GitHub - Keras Model](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py) - ResNet8 implementation (HIGH confidence)
- [MLPerf Tiny v1.3 Announcement (MLCommons, Sep 2025)](https://mlcommons.org/2025/09/mlperf-tiny-v1-3-tech/) - Benchmark details (HIGH confidence)

### Evaluation & Testing
- [torchmetrics GitHub Releases](https://github.com/Lightning-AI/torchmetrics/releases) - Version 1.8.2 (HIGH confidence)
- [PyTorch CIFAR10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) - Updated Sep 2025 (HIGH confidence)
- [PyTorch Lightning CIFAR10 Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html) - 94% accuracy baseline (HIGH confidence)
- [pytest vs unittest for PyTorch (CircleCI)](https://circleci.com/blog/testing-pytorch-model-with-pytest/) - Testing best practices (MEDIUM confidence)

### Compatibility
- [NumPy 2.0 Support - PyTorch Issue #107302](https://github.com/pytorch/pytorch/issues/107302) - NumPy compatibility status (MEDIUM confidence)
- [TensorFlow: Save and Load Models](https://www.tensorflow.org/tutorials/keras/save_and_load) - H5 format documentation (HIGH confidence)

## Confidence Assessment

| Component | Confidence | Rationale |
|-----------|------------|-----------|
| PyTorch/torchvision | HIGH | Official docs, verified latest versions |
| h5py | HIGH | PyPI page, official releases |
| ONNX pipeline | HIGH | All libraries verified on PyPI with recent releases |
| Manual conversion | MEDIUM | Based on community patterns, not official converter |
| NumPy compatibility | MEDIUM | Issue discussions suggest compatibility exists but edge cases possible |
| Testing tools | HIGH | Official PyTorch wiki confirms pytest/unittest support |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Manual conversion errors | Medium | High | Implement numerical verification tests comparing outputs |
| ONNX operator support gaps | Low | Medium | onnx2torch supports ResNet-class models per documentation |
| NumPy version conflicts | Low | Medium | Pin numpy<2 if issues arise |
| Accuracy degradation | Low | High | Target is 85%, Keras model likely exceeds this with margin |
| Layer order mismatch | Medium | High | Use h5py to inspect exact layer names before mapping |

## Next Steps for Implementation

1. **Phase 1**: Inspect .h5 file structure with h5py, document layer names
2. **Phase 2**: Implement PyTorch ResNet8 matching Keras architecture
3. **Phase 3**: Write conversion script with numerical tests
4. **Phase 4**: Implement CIFAR-10 evaluation script
5. **Phase 5**: Validate ≥85% accuracy target

Each phase should have unit tests (pytest) validating correctness before proceeding.
