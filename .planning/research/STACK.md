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

---

# Milestone v1.2: PTQ (Post-Training Quantization) Stack Additions

**Researched:** 2026-01-28
**Confidence:** HIGH

## Executive Summary

**No new dependencies required.** Existing stack (onnxruntime 1.23.2, torch 2.0.0+) already includes PTQ capabilities. Quantization APIs are bundled in base packages.

## Current Stack Analysis (v1.0-v1.1)

Based on `requirements.txt`:
- **onnxruntime** >=1.23.2 - Already includes `onnxruntime.quantization` module
- **torch** >=2.0.0 - Already includes `torch.ao.quantization` module
- **Python** 3.12 - Compatible with all quantization APIs

## PTQ Capabilities: Already Available

### ONNX Runtime Quantization (Built-in)

**Module:** `onnxruntime.quantization` (no installation needed)

**What it provides:**
- `quantize_static()` API for static PTQ
- Three calibration methods: MinMax, Entropy, Percentile
- Support for int8 and uint8 data types
- `CalibrationDataReader` interface for calibration datasets
- Pre-processing utilities via `onnxruntime.quantization.shape_inference`

**Requirements:**
- Model must be ONNX opset 10+ (opset 13+ for per-channel quantization)
- Calibration data: representative CIFAR-10 samples (100-1000 images typical)
- CPU-optimized: int8 inference optimized for x86-64 with VNNI instructions

**Version verified:** ONNX Runtime 1.23.2 released October 2025

**Import pattern:**
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
```

**Sources:**
- [ONNX Runtime Quantization Docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime v1.23.2 Release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2)
- [Quantization Tools README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md)

### PyTorch Quantization (Built-in)

**Module:** `torch.ao.quantization` (no installation needed)

**What it provides:**
- Static quantization in eager mode (beta, stable)
- Observer-based calibration for scale/zero-point computation
- Support for int8 quantization
- Backend: fbgemm (CPU quantization, built into torch on x86)
- QConfig system for specifying quantization schemes

**Requirements:**
- Model must be in eval mode before quantization
- Observers inserted during `prepare()` phase
- Calibration via forward passes with representative CIFAR-10 data
- CPU-only: PyTorch quantization currently supports CPUs only

**Version verified:** PyTorch 2.0.0+ includes stable quantization APIs

**Import pattern:**
```python
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert
```

**Sources:**
- [PyTorch Static Quantization Tutorial](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [PyTorch Quantization API Reference](https://docs.pytorch.org/docs/stable/quantization.html)
- [INT8 Quantization for x86 CPU](https://pytorch.org/blog/int8-quantization/)

## Recommended Stack Changes

### No Changes Required

**Verdict:** Existing `requirements.txt` is sufficient for v1.2 PTQ milestone.

Both onnxruntime and torch ship with quantization APIs as part of their base packages. No additional dependencies needed.

## Optional Dependencies (NOT Recommended for v1.2)

### torchao - DEFERRED

**Library:** torchao (PyTorch Architecture Optimization)
**Version:** 0.15+
**Purpose:** Next-generation PyTorch quantization API

**Why NOT adding now:**
- `torch.ao.quantization` is being deprecated in favor of torchao (PyTorch 2.10+ timeline)
- Current `torch.ao.quantization` APIs still fully functional and documented
- Adds dependency complexity for minimal immediate benefit
- Requires explicit pip install: `pip install torchao`
- Python 3.10+ required (project uses 3.12, compatible)

**Migration timeline:** PyTorch plans to remove `torch.ao.quantization` in version 2.10. Current requirements specify `torch>=2.0.0`, so no immediate pressure.

**Recommendation:** Defer torchao adoption to future milestone when PyTorch 2.10+ becomes minimum version or when current APIs show instability.

**Sources:**
- [torch.ao.quantization Deprecation Tracker](https://github.com/pytorch/ao/issues/2259)
- [torchao PyPI](https://pypi.org/project/torchao/)
- [torchao Documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html)

## Calibration Configuration Recommendations

### ONNX Runtime Static Quantization

**Calibration method options:**
1. **MinMax** - Simple min/max range, fast but can be suboptimal
2. **Entropy** - KL divergence-based, better accuracy retention, slower
3. **Percentile** - Uses percentiles to clip outliers, balanced approach

**Recommendation:** Start with MinMax (fastest iteration), try Entropy if accuracy drop >2%.

**Data type combinations:**
- `(uint8, uint8)` - activations: uint8, weights: uint8 (most common for CNNs)
- `(uint8, int8)` - activations: uint8, weights: int8 (alternative)

**Recommendation:** Use (uint8, uint8) - standard for CNN quantization.

**Calibration dataset size:** 500-1000 random CIFAR-10 training samples (sufficient for statistics).

### PyTorch Static Quantization

**Backend:**
- **fbgemm** - CPU quantization (default), used for x86-64 processors
- **qnnpack** - Mobile/ARM quantization (not needed for this project)

**Recommendation:** Use fbgemm (default) for CPU evaluation.

**Quantization scheme:**
- Per-tensor quantization: simpler, slightly lower accuracy
- Per-channel quantization for weights: better accuracy, requires ONNX opset 13+

**Recommendation:** Per-channel for conv weights, per-tensor for activations (common CNN pattern).

**Calibration dataset size:** Same 500-1000 CIFAR-10 samples (should match ONNX Runtime for fair comparison).

## Integration Points

### Shared Calibration Data

**Critical:** Both frameworks should use identical calibration samples for fair accuracy comparison.

**Implementation approach:**
1. Create `scripts/prepare_calibration.py` - Select and save 500-1000 random CIFAR-10 training images
2. ONNX Runtime: Implement `CalibrationDataReader` subclass reading from saved samples
3. PyTorch: Load same samples for observer calibration during `prepare()` phase

### Model Verification

**ONNX model opset check:**
```python
import onnx
model = onnx.load("models/resnet8.onnx")
print(f"Opset version: {model.opset_import[0].version}")
# Should be >=10, preferably >=13 for per-channel
```

**PyTorch backend check:**
```python
import torch
print(torch.backends.quantized.engine)
# Should print 'fbgemm' on x86-64
```

## Known Constraints

### ONNX Runtime
- **Opset requirement:** Model must be opset 10+ (13+ for per-channel weight quantization)
- **CPU-only:** Quantized int8 inference optimized for x86-64 with VNNI instructions
- **Calibration mandatory:** Static quantization requires calibration data (dynamic quantization does not)

### PyTorch
- **CPU-only:** PyTorch quantization APIs currently only support CPU inference
- **Eager mode limitations:** Requires manual specification of which layers to fuse (conv-bn-relu)
- **Backend availability:** fbgemm backend must be compiled into torch (standard in official binaries)

### Common to Both
- **Accuracy trade-off:** Expect 0-5% accuracy drop from 87.19% baseline (typical for int8 PTQ)
- **Model structure:** Some architectures quantize better than others (ResNet family generally quantizes well)
- **Calibration quality:** Poor calibration data (non-representative) degrades quantized accuracy significantly

## Success Criteria for v1.2

### Functional
1. ONNX Runtime quantization produces valid int8/uint8 ONNX model
2. PyTorch quantization produces valid int8 PyTorch model
3. Both quantized models run inference without errors
4. Accuracy evaluation completes for both frameworks

### Quality
1. Quantized accuracy within 5% of 87.19% baseline (i.e., >82% acceptable, >85% good)
2. Calibration completes in <5 minutes on CPU
3. Inference speedup observed (optional metric, not required for v1.2)

### Documentation
1. Calibration dataset preparation script
2. Quantization scripts for both frameworks
3. Accuracy comparison report (quantized vs baseline)

## Implementation Phases

**Recommended order:**

1. **Phase 1: Calibration data preparation** (shared)
   - Select 500-1000 random CIFAR-10 training samples
   - Save to disk for reproducibility
   - Document selection seed

2. **Phase 2: ONNX Runtime quantization**
   - Verify ONNX model opset version
   - Implement CalibrationDataReader
   - Run quantize_static() with MinMax
   - Evaluate accuracy

3. **Phase 3: PyTorch quantization**
   - Load PyTorch model from .pt file
   - Insert observers with get_default_qconfig()
   - Run calibration with prepare()
   - Convert to quantized model
   - Evaluate accuracy

4. **Phase 4: Analysis**
   - Compare quantized vs baseline accuracy (87.19%)
   - Document accuracy delta
   - If needed, retry with Entropy calibration (ONNX Runtime)

## Sources

### Official Documentation
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Static quantization APIs and calibration methods (HIGH confidence)
- [PyTorch Static Quantization Tutorial](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html) - Eager mode PTQ implementation (HIGH confidence)
- [PyTorch Quantization Overview](https://docs.pytorch.org/docs/stable/quantization.html) - API reference and quantization modes (HIGH confidence)

### Technical Details
- [ONNX Runtime Quantization Tools](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md) - Implementation details (HIGH confidence)
- [ONNX Opset Requirements](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/windows_guides/_ONNX_PTQ_guide.html) - Opset version requirements for quantization (HIGH confidence)
- [PyTorch INT8 Quantization Blog](https://pytorch.org/blog/int8-quantization/) - INT8 quantization performance details (HIGH confidence)

### Version Information
- [ONNX Runtime v1.23.2 Release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) - October 2025 release notes (HIGH confidence)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) - Version history (HIGH confidence)

### Migration Notes (Future Reference)
- [torch.ao.quantization Deprecation Tracker](https://github.com/pytorch/ao/issues/2259) - Deprecation timeline and migration path (HIGH confidence)
- [torchao PyPI](https://pypi.org/project/torchao/) - Next-gen quantization library (HIGH confidence)
- [torchao Quantization Overview](https://docs.pytorch.org/ao/stable/quantization_overview.html) - Future migration target (HIGH confidence)
- [Clarification of PyTorch Quantization Flow Support](https://dev-discuss.pytorch.org/t/clarification-of-pytorch-quantization-flow-support-in-pytorch-and-torchao/2809) - PyTorch 2.0+ quantization roadmap (MEDIUM confidence)

## Confidence Assessment

| Component | Confidence | Rationale |
|-----------|------------|-----------|
| ONNX Runtime quantization | HIGH | Official docs, verified v1.23.2 includes quantization module, widely used |
| PyTorch quantization | HIGH | Official tutorials, torch.ao.quantization stable since PyTorch 1.8+ |
| No new dependencies | HIGH | Both modules verified as built-in to base packages |
| Calibration approach | HIGH | Standard practice documented in official sources |
| Accuracy expectations | MEDIUM | ResNets quantize well typically, but 0-5% drop is estimate based on literature |
| torchao migration timeline | MEDIUM | Deprecation announced but timeline subject to change |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accuracy drop >5% | Low | High | Retry with Entropy calibration, increase calibration samples to 2000 |
| ONNX opset version <10 | Low | High | Reconvert from Keras with newer opset (already using tf2onnx) |
| PyTorch backend unavailable | Very Low | High | fbgemm ships with official torch binaries on x86-64 |
| Calibration data non-representative | Medium | Medium | Use stratified sampling ensuring all 10 classes represented |
| torch.ao.quantization deprecated before v1.2 complete | Very Low | Low | PyTorch 2.10 not released yet, current APIs remain stable |

## Next Steps for v1.2 Implementation

1. **Verify requirements.txt** - Confirm onnxruntime and torch versions satisfy quantization needs (already >=1.23.2 and >=2.0.0)
2. **Check ONNX model opset** - Load `models/resnet8.onnx` and verify opset >=10
3. **Prepare calibration script** - Select 500-1000 CIFAR-10 training samples with seed for reproducibility
4. **Implement ONNX Runtime quantization** - Create `scripts/quantize_onnx.py`
5. **Implement PyTorch quantization** - Create `scripts/quantize_pytorch.py`
6. **Evaluate and compare** - Run both quantized models through evaluation, compare to 87.19% baseline

Each script should follow existing patterns in `scripts/` directory (argparse CLI, clear output logging).
