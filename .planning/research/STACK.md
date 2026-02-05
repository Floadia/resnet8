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

---

# Milestone v1.3: Quantized Operations Documentation Stack

**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

**For documenting quantized operations with visualization:** Three-tool stack using ONNX Python API for programmatic extraction, onnx.tools.net_drawer + pydot for graph visualization, and GitHub-flavored Markdown with MathJax for math equations. **No major new dependencies** - visualization tools require only graphviz (system) and pydot (Python).

## Documentation Objective (v1.3)

Create reference markdown documentation explaining:
1. **QLinear operation math** - Scale, zero-point, integer arithmetic for hardware implementation
2. **QuantizeLinear/DequantizeLinear** - Input/output boundary operations
3. **ONNX graph visualization** - Netron-style diagrams of quantized models
4. **Operation extraction** - Programmatic access to node details, attributes, parameters

## Recommended Stack for v1.3

### ONNX Model Inspection (Already Available)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **onnx** | >=1.17.0 (in pyproject.toml) | ONNX model loading and inspection | Core library - load models, iterate nodes, access attributes |
| **onnxruntime** | >=1.23.2 (in pyproject.toml) | Model validation | Verify model structure before documenting |

**No installation needed** - Already in project dependencies.

**API for extraction:**
```python
import onnx

# Load quantized model
model = onnx.load("models/resnet8_int8.onnx")

# Iterate through nodes
for node in model.graph.node:
    print(f"Op: {node.op_type}")
    print(f"Inputs: {node.input}")
    print(f"Outputs: {node.output}")
    # Access attributes
    for attr in node.attribute:
        print(f"{attr.name}: {attr}")
```

**Confidence:** HIGH - Official ONNX Python API, verified in documentation

**Sources:**
- [ONNX Python API Overview](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)
- [ONNX with Python](https://onnx.ai/onnx/intro/python.html)

### ONNX Graph Visualization (New for v1.3)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **onnx.tools.net_drawer** | Built into onnx | Generate pydot graphs from ONNX models | Official ONNX visualization tool - produces Graphviz DOT representations |
| **pydot** | Latest (>=3.0.0) | Python interface to Graphviz | Convert pydot graphs to PNG/SVG for documentation |
| **graphviz** | Latest (system package) | Graph rendering engine | Renders DOT files to images (PNG, SVG, PDF) |

**Installation:**
```bash
# Python package (only pydot needed, onnx.tools.net_drawer is built-in)
pip install pydot

# System package (graphviz - for rendering)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows (via Chocolatey):
choco install graphviz
```

**Usage pattern:**
```python
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

# Load ONNX model
model = onnx.load("models/resnet8_int8.onnx")

# Generate pydot graph
pydot_graph = GetPydotGraph(
    model.graph,
    name="ResNet8_Quantized",
    rankdir="TB",  # Top-to-bottom layout
    node_producer=GetOpNodeProducer(embed_docstring=False)
)

# Save as DOT file
pydot_graph.write_dot("docs/resnet8_graph.dot")

# Convert to PNG directly
pydot_graph.write_png("docs/resnet8_graph.png")

# Or convert to SVG (better for web docs)
pydot_graph.write_svg("docs/resnet8_graph.svg")
```

**Why NOT Netron directly:**
- Netron is primarily a viewer (GUI application)
- Python API only provides `netron.start()` to launch web viewer
- No programmatic export to images for documentation
- Not designed for automated documentation workflows

**Why NOT onnx-visualizer or onnx-vis:**
- Client-server architectures designed for interactive exploration
- Require running web servers to view visualizations
- More complex than needed for static documentation
- net_drawer is simpler and official ONNX tooling

**Confidence:** HIGH - Official ONNX tools, pydot well-established

**Sources:**
- [ONNX Visualizing a Model Tutorial](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md)
- [onnx.tools.net_drawer Source](https://github.com/onnx/onnx/blob/main/onnx/tools/net_drawer.py)
- [pydot GitHub](https://github.com/pydot/pydot)

### Markdown Math Documentation (Already Available)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **GitHub Markdown** | Native | Documentation format | LaTeX math support via MathJax (May 2024+) |
| **MathJax** | 3.x (GitHub-hosted) | Math equation rendering | Renders LaTeX equations in GitHub markdown |

**No installation needed** - GitHub natively supports math since May 2024.

**Syntax:**
```markdown
## Quantization Formula

Inline math: The quantized value is computed as $q = \text{round}(\frac{x}{s}) + z$

Block equation:
$$
x_{\text{dequantized}} = (q - z_{\text{zero\_point}}) \times s_{\text{scale}}
$$

## QLinearConv Operation

The quantized convolution computes:

$$
\begin{aligned}
Y_{\text{int8}} &= \text{Conv2D}(X_{\text{int8}}, W_{\text{int8}}) \\
Y_{\text{fp32}} &= (Y_{\text{int8}} - z_y) \times s_y
\end{aligned}
$$

Where:
- $X_{\text{int8}}$ is quantized input: $X_{\text{int8}} = \text{round}(X_{\text{fp32}} / s_x) + z_x$
- $W_{\text{int8}}$ is quantized weight
- $s_x, s_y$ are scale factors
- $z_x, z_y$ are zero-points
```

**Rendered output:** GitHub automatically renders LaTeX using MathJax when viewing .md files.

**Alternative for local preview:**
- Use VSCode with extensions: "Markdown Preview Enhanced" or "Markdown All in One"
- Both support LaTeX math rendering in live preview

**Why NOT Jupyter notebooks:**
- Documentation lives in repo as markdown (version controlled, review-friendly)
- Jupyter adds complexity (requires notebook server, JSON format harder to diff)
- Markdown is more accessible and searchable

**Confidence:** HIGH - GitHub native support verified, widely used

**Sources:**
- [GitHub Writing Mathematical Expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- [Math Support in Markdown - GitHub Blog](https://github.blog/news-insights/product-news/math-support-in-markdown/)

### QLinearConv/QLinearMatMul Operation Details (Reference)

**ONNX operator documentation provides:**
- Input/output tensor specifications
- Attribute definitions (dilations, group, kernel_shape, pads, strides)
- Mathematical descriptions
- Type constraints

**Key QLinearConv inputs (for documentation):**
1. `x` (T1): Input tensor [N × C × H × W]
2. `x_scale` (float): Input quantization scale
3. `x_zero_point` (T1): Input quantization zero-point
4. `w` (T2): Weight tensor [M × C/group × kH × kW]
5. `w_scale` (float): Weight quantization scale (scalar or per-channel)
6. `w_zero_point` (T2): Weight quantization zero-point
7. `y_scale` (float): Output quantization scale
8. `y_zero_point` (T3): Output quantization zero-point
9. `B` (optional, T4): Bias tensor [M]

**QLinearMatMul inputs:**
1. `a`, `a_scale`, `a_zero_point` (matrix A)
2. `b`, `b_scale`, `b_zero_point` (matrix B)
3. `y_scale`, `y_zero_point` (output Y)

**Sources:**
- [QLinearConv - ONNX 1.20.0 Documentation](https://onnx.ai/onnx/operators/onnx__QLinearConv.html)
- [QLinearMatMul - ONNX 1.20.0 Documentation](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html)

## Optional Tools (NOT Recommended for v1.3)

### onnx-tool - NOT NEEDED

**Library:** onnx-tool
**Purpose:** Shape inference, MACs/FLOPs counting, subgraph extraction

**Why NOT adding:**
- v1.3 focuses on documentation, not performance profiling
- MACs/FLOPs counting is out of scope (accuracy only, not inference speed)
- ONNX Python API sufficient for operation extraction
- Adds dependency without clear benefit for current milestone

**Defer to:** Future milestone if performance analysis becomes requirement

**Sources:**
- [onnx-tool PyPI](https://pypi.org/project/onnx-tool/0.2.9/)
- [onnx-tool GitHub](https://github.com/ThanatosShinji/onnx-tool)

### Netron - REFERENCE ONLY

**Tool:** Netron (GUI visualizer)
**Purpose:** Interactive ONNX model exploration

**Why NOT using programmatically:**
- Primarily a viewer application (desktop/web GUI)
- Python API limited to `netron.start()` (launches viewer)
- Cannot export visualizations programmatically
- Better as manual reference tool, not for automated docs

**Usage recommendation:** Use Netron manually to explore models and verify your documented visualizations match reality, but use onnx.tools.net_drawer for generating documentation diagrams.

**How to use as reference:**
```bash
# Install (if needed for manual exploration)
pip install netron

# Launch viewer
netron models/resnet8_int8.onnx
```

**Sources:**
- [Netron GitHub](https://github.com/lutzroeder/netron)
- [Netron PyPI](https://pypi.org/project/netron/)

## Implementation Approach for v1.3

### Phase 1: Extract Operation Details Programmatically

**Script:** `scripts/extract_operations.py`

**Purpose:** Parse quantized ONNX models and extract QLinear operation details

**Output:** JSON or structured text file listing:
- All QLinearConv nodes with attributes
- All QLinearMatMul nodes with attributes
- QuantizeLinear/DequantizeLinear nodes at boundaries
- Scale and zero-point values per operation

**Implementation:**
```python
import onnx
import json

def extract_qlinear_ops(model_path):
    model = onnx.load(model_path)
    operations = []

    for node in model.graph.node:
        if node.op_type in ['QLinearConv', 'QLinearMatMul', 'QuantizeLinear', 'DequantizeLinear']:
            op_info = {
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: onnx.helper.get_attribute_value(attr)
                               for attr in node.attribute}
            }
            operations.append(op_info)

    return operations

# Usage
ops = extract_qlinear_ops("models/resnet8_int8.onnx")
with open("docs/operations.json", "w") as f:
    json.dump(ops, f, indent=2, default=str)
```

### Phase 2: Generate Graph Visualizations

**Script:** `scripts/visualize_graph.py`

**Purpose:** Create PNG/SVG visualizations of quantized ONNX models

**Output:**
- `docs/resnet8_int8_graph.png` - Full model graph
- `docs/resnet8_uint8_graph.png` - Alternative quantization
- `docs/qlinear_conv_detail.png` - Zoomed-in single operation

**Implementation:**
```python
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

def visualize_model(model_path, output_path, rankdir="TB"):
    model = onnx.load(model_path)

    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir=rankdir,
        node_producer=GetOpNodeProducer(embed_docstring=False)
    )

    # Save both formats
    pydot_graph.write_png(output_path.replace('.svg', '.png'))
    pydot_graph.write_svg(output_path)
    print(f"Saved visualization to {output_path}")

# Usage
visualize_model("models/resnet8_int8.onnx", "docs/resnet8_int8_graph.svg")
```

### Phase 3: Write Markdown Documentation

**Files to create:**
- `docs/QUANTIZATION_OPERATIONS.md` - Main documentation
- `docs/QLINEAR_CONV.md` - QLinearConv detailed breakdown
- `docs/QLINEAR_MATMUL.md` - QLinearMatMul detailed breakdown
- `docs/QUANTIZE_DEQUANTIZE.md` - Boundary operations

**Structure:**
```markdown
# Quantized Operations Reference

## Overview

This document explains quantized operations in the ResNet8 int8 model for hardware implementation.

## Graph Visualization

![ResNet8 Quantized Graph](resnet8_int8_graph.svg)

## QLinearConv Operation

### Mathematical Formulation

The QLinearConv operation performs quantized convolution:

$$
Y_{\text{quantized}} = \text{Conv2D}(X_{\text{quantized}}, W_{\text{quantized}})
$$

Where quantization is defined as:

$$
X_{\text{quantized}} = \text{clamp}\left(\text{round}\left(\frac{X_{\text{fp32}}}{s_x}\right) + z_x, 0, 255\right)
$$

### Hardware Implementation

For hardware accelerators, the operation decomposes into:

1. **Integer convolution** (int8 × int8 → int32 accumulation)
2. **Requantization** (int32 → int8 with new scale/zero-point)
3. **Activation** (ReLU in quantized domain)

[Detailed parameter table extracted from operations.json]

### ONNX Node Structure

```
Node: QLinearConv_0
  Inputs: ['x_quantized', 'x_scale', 'x_zero_point',
           'w_quantized', 'w_scale', 'w_zero_point',
           'y_scale', 'y_zero_point', 'bias']
  Attributes:
    - dilations: [1, 1]
    - group: 1
    - kernel_shape: [3, 3]
    - pads: [1, 1, 1, 1]
    - strides: [1, 1]
```
```

## Documentation Checklist for v1.3

- [ ] Extract all QLinearConv nodes from resnet8_int8.onnx
- [ ] Extract all QLinearMatMul nodes (if any)
- [ ] Identify QuantizeLinear/DequantizeLinear boundaries
- [ ] Generate full graph visualizations (PNG + SVG)
- [ ] Write mathematical formulations with LaTeX
- [ ] Document hardware implementation considerations
- [ ] Create parameter reference tables
- [ ] Cross-reference with official ONNX operator docs
- [ ] Verify equations render correctly on GitHub
- [ ] Test local markdown preview (VSCode)

## Stack Summary for v1.3

| Component | Technology | Installation | Confidence |
|-----------|-----------|--------------|------------|
| **Model loading** | onnx (>=1.17.0) | Already installed | HIGH |
| **Operation extraction** | onnx.helper API | Built into onnx | HIGH |
| **Graph visualization** | onnx.tools.net_drawer | Built into onnx | HIGH |
| **DOT to image** | pydot + graphviz | `pip install pydot` + system graphviz | HIGH |
| **Math equations** | GitHub Markdown + MathJax | No install (native) | HIGH |

## Installation for v1.3

### Minimal Addition to Existing Stack

```bash
# Only one new Python package needed
pip install pydot

# System dependency (graphviz)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows:
choco install graphviz
# OR download from https://graphviz.org/download/
```

### Verify Installation

```bash
# Test pydot import
python -c "import pydot; print('pydot OK')"

# Test graphviz command-line tool
dot -V
# Should output: dot - graphviz version X.X.X

# Test onnx.tools.net_drawer
python -c "from onnx.tools.net_drawer import GetPydotGraph; print('net_drawer OK')"
```

## Integration with Existing Project

### Current project structure:
```
resnet8/
├── models/
│   ├── resnet8_int8.pt       # PyTorch quantized model (v1.2)
│   └── resnet8.pt             # PyTorch fp32 model (v1.1)
├── scripts/
│   ├── quantize_onnx.py      # ONNX quantization (v1.2)
│   └── quantize_pytorch.py   # PyTorch quantization (v1.2)
└── pyproject.toml            # Dependencies
```

### Additions for v1.3:
```
resnet8/
├── docs/                     # NEW - Documentation directory
│   ├── QUANTIZATION_OPERATIONS.md
│   ├── QLINEAR_CONV.md
│   ├── resnet8_int8_graph.png
│   └── operations.json
└── scripts/
    ├── extract_operations.py  # NEW - Extract op details
    └── visualize_graph.py     # NEW - Generate diagrams
```

### Workflow:
1. Run `scripts/extract_operations.py` → `docs/operations.json`
2. Run `scripts/visualize_graph.py` → `docs/*.png` and `docs/*.svg`
3. Write markdown docs referencing extracted data and diagrams
4. Push to GitHub → Math equations render automatically

## Known Constraints

### Graphviz System Dependency
- **Issue:** graphviz must be installed at system level (not just Python package)
- **Why:** pydot calls `dot` command-line tool to render graphs
- **Mitigation:** Document installation for common platforms (apt, brew, choco)
- **Fallback:** If graphviz unavailable, can still generate .dot files and render elsewhere

### Large Graph Complexity
- **Issue:** ResNet8 graph may be too large for readable single-image visualization
- **Why:** Many nodes and edges create cluttered diagrams
- **Mitigation:**
  1. Use SVG format (zoomable in browsers)
  2. Generate subgraph visualizations (extract specific layers)
  3. Adjust rankdir ("TB" vs "LR") for better layout
  4. Use Netron manually for interactive exploration, export screenshots

### GitHub Math Rendering
- **Issue:** Math only renders on GitHub web interface, not in git clients or some editors
- **Why:** Requires MathJax JavaScript processing
- **Mitigation:** Use VSCode extensions for local preview during writing

## Success Criteria for v1.3 Documentation

### Functional
1. All quantized models visualized as PNG/SVG diagrams
2. All QLinear operations extracted with complete parameter lists
3. Markdown documentation renders math equations correctly on GitHub
4. Scripts run without errors on clean Python environment

### Quality
1. Documentation explains hardware-implementable integer arithmetic
2. Mathematical formulations are precise and verifiable
3. Visualizations are readable (not cluttered)
4. Cross-references to official ONNX operator documentation provided

### Completeness
1. QLinearConv operations fully documented
2. QuantizeLinear/DequantizeLinear boundary operations explained
3. Scale and zero-point propagation through graph documented
4. Example implementation pseudocode provided for hardware developers

## Sources Summary

### Official ONNX Documentation (HIGH confidence)
- [ONNX Python API Overview](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)
- [ONNX with Python](https://onnx.ai/onnx/intro/python.html)
- [QLinearConv Operator](https://onnx.ai/onnx/operators/onnx__QLinearConv.html)
- [QLinearMatMul Operator](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html)

### Visualization Tools (HIGH confidence)
- [ONNX Visualizing a Model Tutorial](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md)
- [onnx.tools.net_drawer Source Code](https://github.com/onnx/onnx/blob/main/onnx/tools/net_drawer.py)
- [pydot GitHub Repository](https://github.com/pydot/pydot)

### Markdown Math Support (HIGH confidence)
- [GitHub Writing Mathematical Expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- [Math Support in Markdown - GitHub Blog](https://github.blog/news-insights/product-news/math-support-in-markdown/)

### Reference Tools (MEDIUM confidence - supplementary)
- [Netron GitHub](https://github.com/lutzroeder/netron) - Manual exploration tool
- [onnx-tool PyPI](https://pypi.org/project/onnx-tool/0.2.9/) - Optional profiling (deferred)

## Risk Assessment for v1.3

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Graphviz not installed | Medium | High | Clear installation docs for all platforms, check in scripts |
| Graph too complex to visualize | Medium | Low | Use SVG (zoomable), generate subgraphs, provide Netron as fallback |
| Math rendering issues | Low | Medium | Test on GitHub before finalizing, provide VSCode preview instructions |
| Missing ONNX models | Low | High | v1.2 already generated int8/uint8 ONNX models - verify they exist |
| Operation extraction incomplete | Low | Medium | Cross-reference with Netron manual inspection |

## Next Steps for v1.3 Implementation

1. **Install pydot + graphviz** - Add to dev dependencies, document system install
2. **Create docs/ directory** - Initialize documentation structure
3. **Write extract_operations.py** - Parse ONNX models, output JSON
4. **Write visualize_graph.py** - Generate PNG/SVG diagrams
5. **Draft QUANTIZATION_OPERATIONS.md** - Main documentation with math equations
6. **Generate all visualizations** - Run visualization script on all quantized models
7. **Validate on GitHub** - Push and verify math rendering, image display
8. **Cross-reference official docs** - Link to ONNX operator documentation

Each script follows existing project patterns: argparse CLI, clear logging, error handling.

---

# Milestone v1.4: Quantization Playground Stack Additions

**Researched:** 2026-02-05
**Confidence:** HIGH

## Executive Summary

The existing stack (PyTorch, ONNX Runtime, onnx2torch) provides the quantization foundation. This milestone identifies **minimal additions** needed for interactive experimentation: **Marimo** for reactive notebooks and **Plotly** for visualization.

**Key insight:** The project already has `extract_operations.py` which extracts quantization parameters from ONNX models. The playground should leverage this existing infrastructure rather than adding heavy dependencies like onnx-graphsurgeon.

## Recommended Stack Additions for v1.4

### Interactive Notebook

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **marimo** | >=0.19.7 | Reactive notebook environment | Pure Python files (git-friendly), reactive execution, built-in widgets. No callback boilerplate. |

**Version rationale:** v0.19.7 (Jan 29, 2026) is current stable with PDF export and improved sandboxing. Requires Python >=3.10; project uses 3.12.

**Why marimo over Jupyter:**
- Pure Python files (`.py`) - git-friendly, easy to diff and review
- Reactive execution - slider changes automatically re-run dependent cells
- Built-in UI widgets (`mo.ui.slider`, `mo.ui.table`) - no ipywidgets complexity
- Deployable as web apps with zero changes
- No hidden state issues - DAG-based execution model

**Source:** [PyPI marimo](https://pypi.org/project/marimo/) - version verified 2026-02-05

### Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **plotly** | >=6.5.0 | Interactive heatmaps, charts | Native marimo integration via `mo.ui.plotly()`. Heatmaps ideal for visualizing weight/activation tensors. |

**Version rationale:** v6.5.2 (Jan 14, 2026) is current. v6.x has improved performance and supports Python 3.8-3.13.

**Why Plotly over matplotlib:**
- Matplotlib plots are **not reactive** in marimo
- Plotly integrates with marimo's reactive model - automatic updates when sliders change
- `mo.ui.plotly()` enables selection callbacks (click on heatmap cell to inspect values)
- Better suited for tensor heatmaps with zoom/pan/hover

**Why Plotly over Altair:**
- Altair excels at statistical charts but Plotly's `px.imshow()` is more mature for 2D tensor visualization
- Plotly heatmaps support annotation text, custom colorscales, aspect ratio control
- Both work with marimo, but Plotly better matches this use case

**Source:** [PyPI plotly](https://pypi.org/project/plotly/) - version verified 2026-02-05

### Supporting Libraries (Already in Stack - No Changes)

| Library | Current Version | Playground Use |
|---------|-----------------|----------------|
| onnx | >=1.17.0 | Load models, extract graph structure |
| onnxruntime | >=1.23.2 | Run inference, capture outputs |
| torch | >=2.0.0 | PyTorch quantized model inspection |
| numpy | >=1.26.4 | Tensor manipulation |

**No version changes needed** for existing dependencies.

## What NOT to Add

| Library | Why Not |
|---------|---------|
| **onnx-graphsurgeon** | Heavy NVIDIA dependency. Existing `extract_operations.py` already handles parameter extraction using standard `onnx` APIs. |
| **altair** | Plotly better suited for tensor heatmaps. Adding both creates confusion for which to use. |
| **ipywidgets** | Marimo has its own widget system (`mo.ui`). ipywidgets are Jupyter-specific. |
| **jupyter/jupyterlab** | Marimo is the chosen notebook environment. Don't mix notebook systems. |
| **tensorboard** | Overkill for this use case. Plotly heatmaps sufficient for weight visualization. |
| **netron** | External GUI tool. `visualize_graph.py` already exists for static diagrams; notebook doesn't need another viewer. |
| **sklearn-onnx helpers** | Only needed for intermediate output extraction pattern; we'll implement our own simpler approach. |

## Integration Patterns

### ONNX Model Inspection

Reuse existing infrastructure:

```python
# Existing: scripts/extract_operations.py
# Extracts: QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear
# Returns: scales, zero_points, attributes per node

# In notebook:
import sys
sys.path.insert(0, "/path/to/scripts")
from extract_operations import extract_qlinear_operations

ops = extract_qlinear_operations("models/resnet8_int8.onnx")
# ops["operations"] contains all QLinear nodes with scales/zero_points
```

**Confidence:** HIGH - existing working code in project

### ONNX Intermediate Tensor Capture

ONNX Runtime does **not** support direct intermediate output extraction. Standard approach:

```python
import onnx
from onnx import helper
import onnxruntime as ort

def add_intermediate_output(model_path, tensor_name, output_path):
    """Modify ONNX model to expose an intermediate tensor as output."""
    model = onnx.load(model_path)

    # Add the intermediate tensor to graph outputs
    intermediate = helper.make_value_info(tensor_name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(intermediate)

    onnx.save(model, output_path)
    return output_path

# Run inference with modified model
modified_path = add_intermediate_output("model.onnx", "layer3_output", "model_debug.onnx")
sess = ort.InferenceSession(modified_path)
outputs = sess.run(None, {"input": input_data})
# outputs now includes the intermediate tensor
```

**Confidence:** HIGH - verified via [sklearn-onnx documentation](http://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html)

### PyTorch Intermediate Tensor Capture

Use forward hooks (standard PyTorch API):

```python
activations = {}

def make_hook(name):
    def hook_fn(module, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook_fn

# Register hooks on quantized model
model = torch.jit.load("models/resnet8_int8.pt")
for name, module in model.named_modules():
    if "conv" in name or "linear" in name:  # Filter to layers of interest
        module.register_forward_hook(make_hook(name))

# Run inference
with torch.no_grad():
    _ = model(input_tensor)

# activations dict now contains all intermediate outputs
```

**Confidence:** HIGH - standard PyTorch API, verified via [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html)

**Note for JIT models:** PyTorch JIT-traced models may have limited hook support. If FX-quantized model hooks fail, the fallback is to load the pre-traced model and add hooks before tracing.

### PyTorch Quantization Parameter Inspection

Quantized tensors store scale/zero_point directly:

```python
def extract_quant_params(model):
    """Extract quantization parameters from PyTorch quantized model."""
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and callable(module.weight):
            try:
                w = module.weight()
                if w.is_quantized:
                    params[name] = {
                        "scale": w.q_scale(),
                        "zero_point": w.q_zero_point(),
                        "dtype": str(w.dtype),
                        "shape": list(w.shape)
                    }
            except Exception:
                pass  # Skip modules without quantized weights
    return params
```

**Confidence:** HIGH - standard `torch.ao.quantization` API

### Marimo UI Integration

```python
import marimo as mo
import plotly.express as px
import pandas as pd

# Reactive slider for scale modification
scale_slider = mo.ui.slider(
    start=0.001, stop=0.1, step=0.001,
    value=0.01,
    label="Scale factor",
    show_value=True
)

# Table for displaying quantization parameters
ops_df = pd.DataFrame([
    {"layer": op["name"], "scale": op["scales"].get("x_scale", "N/A"), ...}
    for op in ops["operations"]
])
params_table = mo.ui.table(ops_df, selection="single")

# Plotly heatmap for tensor visualization (reactive with marimo)
weight_tensor = np.array(...)  # Shape: (out_channels, in_channels, H, W)
# Reshape to 2D for visualization
weight_2d = weight_tensor.reshape(weight_tensor.shape[0], -1)

fig = px.imshow(
    weight_2d,
    color_continuous_scale="RdBu",
    aspect="auto",
    labels={"x": "Flattened spatial/channel", "y": "Output channel"}
)
mo.ui.plotly(fig)  # Reactive plot - updates when dependent data changes
```

**Confidence:** HIGH - verified via [marimo interactivity docs](https://docs.marimo.io/guides/interactivity/)

## Installation

### Add to pyproject.toml

```toml
[project]
dependencies = [
    # ... existing deps ...
    "marimo>=0.19.7",
    "plotly>=6.5.0",
]
```

### Direct install

```bash
# Using uv (recommended for this project)
uv add marimo plotly

# Or using pip
pip install "marimo>=0.19.7" "plotly>=6.5.0"
```

## Development Workflow

```bash
# Start marimo editor (opens browser)
marimo edit notebooks/quantization_playground.py

# Or edit with VS Code extension
code notebooks/quantization_playground.py
# Then: Ctrl+Shift+P -> "Marimo: Open in browser"

# Run as script (for CI/testing)
python notebooks/quantization_playground.py

# Export to static HTML (for sharing)
marimo export html notebooks/quantization_playground.py -o playground.html

# Export to PDF
marimo export pdf notebooks/quantization_playground.py -o playground.pdf
```

## Project Structure Additions

```
resnet8/
├── notebooks/                    # NEW - Marimo notebooks
│   ├── quantization_playground.py  # Main interactive playground
│   └── __init__.py               # (optional) for imports
├── scripts/
│   ├── extract_operations.py     # (existing) - reuse in notebook
│   ├── ...
└── pyproject.toml               # Add marimo, plotly
```

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| marimo version | HIGH | Verified via PyPI 2026-02-05, v0.19.7 current |
| plotly version | HIGH | Verified via PyPI 2026-02-05, v6.5.2 current |
| marimo-plotly integration | HIGH | Documented `mo.ui.plotly()` API in official docs |
| ONNX intermediate capture | HIGH | sklearn-onnx docs confirm approach; standard pattern |
| PyTorch hooks | HIGH | Standard torch API, well documented |
| Quantization param access | MEDIUM | Standard API but JIT models may differ from eager mode |

## Open Questions for Implementation

1. **JIT model hook compatibility:** PyTorch JIT-traced models (from FX quantization) may have limited hook support. Needs validation during implementation - may need to capture activations before JIT tracing.

2. **Parameter modification persistence:** Modifying scale/zero_point in-memory is straightforward; saving modified models back to disk requires careful reconstruction of the ONNX graph or PyTorch state dict.

3. **Large tensor visualization:** For high-dimensional weights (e.g., 64x64x3x3 conv), need strategy for reducing to 2D display (mean over channels, PCA, etc.).

## Sources

### Version Information (HIGH confidence)
- [marimo PyPI](https://pypi.org/project/marimo/) - version 0.19.7 verified 2026-02-05
- [plotly PyPI](https://pypi.org/project/plotly/) - version 6.5.2 verified 2026-02-05

### marimo Documentation (HIGH confidence)
- [marimo interactivity guide](https://docs.marimo.io/guides/interactivity/)
- [marimo plotting guide](https://docs.marimo.io/guides/working_with_data/plotting/)
- [marimo slider API](https://docs.marimo.io/api/inputs/slider/)
- [marimo table API](https://docs.marimo.io/api/inputs/table/)

### Intermediate Tensor Capture (HIGH confidence)
- [sklearn-onnx intermediate outputs](http://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html)
- [PyTorch forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html)

### Quantization APIs (HIGH confidence)
- [PyTorch quantization docs](https://docs.pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
