# Architecture: Marimo Quantization Playground Integration

**Project:** ResNet8 CIFAR-10 Quantization Playground
**Milestone:** v1.4 - Interactive Quantization Playground
**Researched:** 2026-02-05
**Confidence:** HIGH

## Executive Summary

The Marimo quantization playground should integrate with existing ResNet8 scripts as a thin interactive layer, NOT by reimplementing functionality. The architecture follows a **wrapper pattern**: Marimo notebooks import and call existing utility functions from `scripts/`, display results interactively, and allow parameter modification through reactive UI elements.

**Key architectural decision:** Refactor existing scripts into importable modules with clear entry points, then build Marimo notebooks that compose these modules with interactive controls.

---

## Existing Component Inventory

### Models (in `models/`)

| File | Format | Content | Access Pattern |
|------|--------|---------|----------------|
| `resnet8.pt` | PyTorch checkpoint | FP32 model (353 KB) | `torch.load()` with `weights_only=False` |
| `resnet8_int8.pt` | TorchScript | INT8 quantized model (169 KB) | `torch.jit.load()` |
| `resnet8_int8_operations.json` | JSON | Extracted QDQ operations with scales/zero-points | `json.load()` |

**Note:** ONNX models (`resnet8.onnx`, `resnet8_int8.onnx`) may be generated on-demand but are not committed to repo.

### Scripts (in `scripts/`)

| Script | Functions | Reuse Potential |
|--------|-----------|-----------------|
| `evaluate_pytorch.py` | `load_cifar10_test()`, `load_pytorch_model()`, `evaluate_model()`, `compute_accuracy()` | HIGH - direct import |
| `quantize_pytorch.py` | `load_pytorch_model()`, `quantize_model_fx()`, `create_calibration_loader()` | MEDIUM - needs refactoring |
| `calibration_utils.py` | `load_calibration_data()`, `verify_distribution()` | HIGH - already module-friendly |
| `extract_operations.py` | `extract_qlinear_operations()` | HIGH - returns structured dict |
| `visualize_graph.py` | Graph visualization utilities | HIGH - for model structure display |
| `annotate_qdq_graph.py` | QDQ architecture diagrams | MEDIUM - generates static images |

### Documentation (in `docs/quantization/`)

| File | Content | Playground Use |
|------|---------|----------------|
| `01-boundary-operations.md` | QuantizeLinear/DequantizeLinear formulas | Reference in UI tooltips |
| `02-qlinearconv.md` | QLinearConv operation details | Context for conv layer inspection |
| `03-qlinearmatmul.md` | QLinearMatMul operation details | Context for dense layer inspection |
| `04-architecture.md` | QDQ format architecture overview | High-level understanding |

---

## Proposed Architecture

### Component Diagram

```
+------------------------------------------------------------------+
|                      Marimo Notebook Layer                        |
|  +------------------+  +------------------+  +-----------------+  |
|  | Model Inspector  |  | Parameter Editor |  | Comparison View |  |
|  | (mo.ui.tree,     |  | (mo.ui.slider,   |  | (mo.ui.table,   |  |
|  |  mo.ui.table)    |  |  mo.ui.number)   |  |  plots)         |  |
|  +--------+---------+  +--------+---------+  +--------+--------+  |
|           |                     |                     |           |
+-----------+---------------------+---------------------+-----------+
            |                     |                     |
            v                     v                     v
+------------------------------------------------------------------+
|                    Playground Utilities Layer                     |
|  playground/                                                      |
|  +------------------+  +------------------+  +-----------------+  |
|  | model_loader.py  |  | param_editor.py  |  | comparison.py   |  |
|  | - load_model()   |  | - modify_scale() |  | - run_compare() |  |
|  | - get_params()   |  | - modify_zp()    |  | - diff_output() |  |
|  +--------+---------+  +--------+---------+  +--------+--------+  |
|           |                     |                     |           |
+-----------+---------------------+---------------------+-----------+
            |                     |                     |
            v                     v                     v
+------------------------------------------------------------------+
|                    Existing Scripts Layer                         |
|  scripts/                                                         |
|  +------------------+  +------------------+  +-----------------+  |
|  | calibration_     |  | evaluate_        |  | extract_        |  |
|  | utils.py         |  | pytorch.py       |  | operations.py   |  |
|  +------------------+  +------------------+  +-----------------+  |
+------------------------------------------------------------------+
            |                     |                     |
            v                     v                     v
+------------------------------------------------------------------+
|                         Data Layer                                |
|  models/                           External Data                  |
|  +------------------+             +------------------+            |
|  | resnet8.pt       |             | CIFAR-10 dataset |            |
|  | resnet8_int8.pt  |             | (calibration/    |            |
|  | *_operations.json|             |  test batches)   |            |
|  +------------------+             +------------------+            |
+------------------------------------------------------------------+
```

### Integration Points

#### 1. Model Loading Integration

**Existing code:** `scripts/evaluate_pytorch.py::load_pytorch_model()`

```python
# Current implementation handles both formats:
def load_pytorch_model(model_path: str) -> torch.nn.Module:
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except RuntimeError:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = checkpoint["model"]
        model.eval()
        return model
```

**Marimo integration:**

```python
# playground/model_loader.py
import sys
sys.path.insert(0, "scripts")
from evaluate_pytorch import load_pytorch_model

def load_with_metadata(model_path: str) -> dict:
    """Load model and extract metadata for UI display."""
    model = load_pytorch_model(model_path)
    return {
        "model": model,
        "path": model_path,
        "type": "TorchScript" if hasattr(model, "graph") else "Checkpoint",
        "is_quantized": "int8" in model_path.lower(),
    }
```

#### 2. Parameter Extraction Integration

**Existing code:** `scripts/extract_operations.py::extract_qlinear_operations()`

```python
# Returns structured dict with scales and zero-points per operation
{
    "model_path": "...",
    "opset_version": 15,
    "operations": [
        {
            "name": "...",
            "op_type": "DequantizeLinear",
            "scales": {"...": 0.026842},
            "zero_points": {"...": -79}
        },
        ...
    ],
    "summary": {"total_nodes": 130, "qlinear_nodes": 98}
}
```

**Marimo integration:**

```python
# playground/param_inspector.py
import json

def load_operations(json_path: str) -> dict:
    """Load pre-extracted operations for interactive inspection."""
    with open(json_path) as f:
        return json.load(f)

def get_layer_params(operations: dict, layer_name: str) -> dict:
    """Extract scale/zero-point for specific layer."""
    for op in operations["operations"]:
        if layer_name in op["name"]:
            return {
                "name": op["name"],
                "op_type": op["op_type"],
                "scales": op.get("scales", {}),
                "zero_points": op.get("zero_points", {})
            }
    return None
```

#### 3. Inference Integration

**Existing code:** `scripts/evaluate_pytorch.py::evaluate_model()`

**New capability needed:** Capture intermediate layer outputs.

```python
# playground/inference.py
import torch

def run_inference_with_hooks(model, images, capture_layers=None):
    """Run inference and capture intermediate activations."""
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().clone()
        return hook

    # Register hooks for named modules
    for name, module in model.named_modules():
        if capture_layers is None or name in capture_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run inference
    images_tensor = torch.from_numpy(images)
    with torch.no_grad():
        outputs = model(images_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return outputs, activations
```

#### 4. Calibration Data Integration

**Existing code:** `scripts/calibration_utils.py::load_calibration_data()`

**Marimo integration:** Direct import, no changes needed.

```python
# In Marimo notebook
import sys
sys.path.insert(0, "scripts")
from calibration_utils import load_calibration_data

# Load subset for quick experiments
images, labels, class_names = load_calibration_data(
    data_dir="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
    samples_per_class=10  # Small subset for interactive use
)
```

---

## Data Flow Architecture

### Flow 1: Model Inspection

```
User selects model file (mo.ui.dropdown)
    |
    v
load_with_metadata(path) --> Model object + metadata dict
    |
    v
Extract state_dict or graph structure
    |
    v
Display in mo.ui.tree (hierarchical layer view)
    |
    v
User clicks layer --> Show layer details (shape, dtype, params)
```

### Flow 2: Parameter Inspection (ONNX-based)

```
Load pre-extracted operations.json
    |
    v
Display operations in mo.ui.table (sortable, filterable)
    |
    v
User selects operation --> Show scale/zero-point values
    |
    v
Display quantization range visualization (mo.ui.plotly or altair)
    |
    v
User modifies slider --> Re-quantize and show effect
```

### Flow 3: Inference Comparison

```
Select original model (FP32) and quantized model (INT8)
    |
    v
Load sample images from calibration data
    |
    v
Run inference on both models with hooks
    |
    v
Capture intermediate activations
    |
    v
Display side-by-side comparison:
- Per-layer output differences
- Final prediction differences
- Accuracy metrics
```

### Flow 4: Parameter Modification Experiment

```
Load quantized PyTorch model
    |
    v
Display editable scale/zero-point parameters
    |
    v
User modifies parameter via slider/input
    |
    +---> [Temporary modification in memory]
    |
    v
Re-run inference with modified parameters
    |
    v
Compare to original quantized output:
- Output tensor diff
- Accuracy impact
- Per-class accuracy changes
```

---

## New Components to Build

### 1. `playground/` Package

New directory for Marimo-specific utilities.

```
playground/
├── __init__.py           # Package exports
├── model_loader.py       # Unified model loading
├── param_inspector.py    # Parameter extraction and display
├── inference.py          # Inference with activation capture
├── comparison.py         # Model comparison utilities
└── visualization.py      # Plotting helpers for Marimo
```

### 2. Marimo Notebooks

```
notebooks/
├── 01_model_inspector.py      # Browse model structure
├── 02_parameter_viewer.py     # View quantization params
├── 03_inference_comparison.py # Compare FP32 vs INT8
├── 04_parameter_editor.py     # Modify params interactively
└── 05_full_playground.py      # Combined experience
```

### 3. Script Refactoring (Minimal)

**Goal:** Make existing scripts importable without running `main()`.

**Current issue:** Some scripts have side effects at import time (e.g., adding to sys.path).

**Fix pattern:**

```python
# Before
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_utils import load_calibration_data

# After (move to function)
def get_calibration_loader(...):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibration_utils import load_calibration_data
    # ...
```

---

## Build Order Rationale

### Phase 1: Model Inspection (Foundation)

**Components:**
- `playground/model_loader.py`
- `notebooks/01_model_inspector.py`

**Why first:**
- Simplest integration (just loading and displaying)
- Validates Marimo + PyTorch interop
- No parameter modification complexity
- Users can immediately explore model structure

**Dependencies:** None (uses existing `evaluate_pytorch.py`)

### Phase 2: Parameter Viewing (Read-Only)

**Components:**
- `playground/param_inspector.py`
- `notebooks/02_parameter_viewer.py`

**Why second:**
- Builds on model loading
- Still read-only (no modifications)
- Validates JSON data flow
- Foundation for parameter editing

**Dependencies:** Phase 1, existing `operations.json`

### Phase 3: Inference with Hooks

**Components:**
- `playground/inference.py`
- `notebooks/03_inference_comparison.py`

**Why third:**
- Needs activation capture (new capability)
- More complex than viewing
- Foundation for comparison features

**Dependencies:** Phase 1, calibration data access

### Phase 4: Parameter Modification

**Components:**
- `playground/param_editor.py`
- `notebooks/04_parameter_editor.py`

**Why fourth:**
- Most complex (modifying model state)
- Needs inference to validate changes
- Experimental feature

**Dependencies:** Phases 1-3

### Phase 5: Integrated Playground

**Components:**
- `playground/comparison.py`
- `notebooks/05_full_playground.py`

**Why last:**
- Combines all features
- Needs all components working
- Polish and UX refinement

**Dependencies:** Phases 1-4

---

## Technical Decisions

### Decision 1: PyTorch Primary, ONNX Secondary

**Recommendation:** Focus on PyTorch model inspection.

**Rationale:**
- PyTorch TorchScript models are already in `models/`
- PyTorch allows hook-based activation capture
- Parameter modification is more straightforward
- ONNX inspection via `operations.json` (pre-extracted)

**ONNX access:** Use pre-extracted JSON for parameter viewing rather than loading ONNX models directly. This avoids ONNX Runtime dependency complexity in Marimo.

### Decision 2: Reactive Parameter Modification

**Recommendation:** Use Marimo's reactive execution for instant feedback.

**Pattern:**

```python
import marimo as mo

# Slider bound to variable
scale_slider = mo.ui.slider(0.001, 0.1, value=0.03, step=0.001)

# Cell that depends on slider
@mo.reactive
def modified_inference():
    modified_scale = scale_slider.value
    # Modify model parameter
    # Run inference
    # Return results
    return results
```

**Rationale:** Marimo's reactive model eliminates callback boilerplate and ensures consistent state.

### Decision 3: Project Notebook Pattern

**Recommendation:** Use Marimo as project notebook (not sandbox).

**Configuration in `pyproject.toml`:**

```toml
[dependency-groups]
dev = [
    "ruff>=0.8.0",
    "marimo>=0.10.0",  # Add marimo as dev dependency
]
```

**Rationale:**
- Shares dependencies with project (torch, onnx, etc.)
- Can import from `scripts/` directly
- No inline dependency management needed

### Decision 4: Minimal Script Modification

**Recommendation:** Prefer wrapping over modifying existing scripts.

**Pattern:**

```python
# playground/model_loader.py
def load_model_for_playground(path):
    """Wrapper that handles playground-specific concerns."""
    import sys
    import os

    # Temporarily modify path
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    sys.path.insert(0, scripts_dir)

    try:
        from evaluate_pytorch import load_pytorch_model
        return load_pytorch_model(path)
    finally:
        sys.path.remove(scripts_dir)
```

**Rationale:**
- Existing scripts continue working standalone
- Playground has its own import handling
- Easier to maintain both use cases

---

## Marimo UI Component Mapping

| Feature | Marimo Component | Usage |
|---------|-----------------|-------|
| Model selection | `mo.ui.dropdown` | Select from available models |
| Layer tree | `mo.ui.tree` | Hierarchical model structure |
| Parameter table | `mo.ui.table` | Scale/zero-point display |
| Scale editor | `mo.ui.slider` + `mo.ui.number` | Modify scale values |
| Image viewer | `mo.image` | Show sample images |
| Activation heatmap | plotly/altair + `mo.ui.plotly` | Visualize activations |
| Comparison view | `mo.hstack`/`mo.vstack` | Side-by-side layouts |
| Metrics display | `mo.stat` | Accuracy/loss numbers |
| Layer selection | `mo.ui.multiselect` | Choose layers to inspect |

---

## File Structure After Implementation

```
resnet8/
├── models/
│   ├── resnet8.pt
│   ├── resnet8_int8.pt
│   └── resnet8_int8_operations.json
├── scripts/
│   ├── calibration_utils.py      # Unchanged
│   ├── evaluate_pytorch.py       # Unchanged
│   ├── extract_operations.py     # Unchanged
│   └── ...
├── playground/                    # NEW
│   ├── __init__.py
│   ├── model_loader.py
│   ├── param_inspector.py
│   ├── inference.py
│   ├── comparison.py
│   └── visualization.py
├── notebooks/                     # NEW
│   ├── 01_model_inspector.py
│   ├── 02_parameter_viewer.py
│   ├── 03_inference_comparison.py
│   ├── 04_parameter_editor.py
│   └── 05_full_playground.py
├── pyproject.toml                 # Add marimo to dev deps
└── ...
```

---

## Risk Mitigation

### Risk 1: TorchScript Quantized Model Modification

**Challenge:** TorchScript models are frozen; parameters may not be easily modifiable.

**Mitigation:**
1. For viewing: Extract parameters without modification (works)
2. For editing: May need to re-quantize from FP32 with modified parameters
3. Alternative: Load checkpoint format instead of TorchScript where possible

### Risk 2: Hook Compatibility with Quantized Models

**Challenge:** PyTorch hooks may not capture quantized tensor internals correctly.

**Mitigation:**
1. Test hooks on both FP32 and INT8 models early
2. Use TorchScript graph inspection as fallback
3. Document which layers support hook-based inspection

### Risk 3: Large Model/Data in Interactive Environment

**Challenge:** Full CIFAR-10 (10K images) too slow for interactive feedback.

**Mitigation:**
1. Use calibration subset (100-1000 images) for experiments
2. Cache inference results where possible
3. Provide progress indicators for longer operations

---

## Sources

### Marimo Documentation (HIGH confidence)
- [Marimo Official Docs](https://docs.marimo.io/) - API reference and guides
- [Marimo Best Practices](https://docs.marimo.io/guides/best_practices/) - Notebook organization
- [Notebooks in Projects](https://docs.marimo.io/guides/package_management/notebooks_in_projects/) - Integration patterns

### PyTorch Documentation (HIGH confidence)
- [PyTorch Hooks Tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) - Activation capture
- [TorchScript Reference](https://pytorch.org/docs/stable/jit.html) - Scripted model inspection

### Project-Specific (HIGH confidence)
- Existing scripts in `scripts/` directory - Direct code review
- `models/resnet8_int8_operations.json` - Pre-extracted quantization parameters
- `docs/quantization/04-architecture.md` - QDQ format understanding

---

## Confidence Assessment

| Component | Confidence | Rationale |
|-----------|------------|-----------|
| Model loading integration | HIGH | Existing code works, wrapper is straightforward |
| Parameter extraction | HIGH | JSON format already defined and working |
| Marimo reactive UI | HIGH | Well-documented, standard patterns |
| Hook-based activation capture | MEDIUM | PyTorch hooks work but quantized model behavior needs testing |
| Parameter modification | MEDIUM | TorchScript limitations may require workarounds |
| Full integration | MEDIUM | Complex composition, needs iterative refinement |

---

## Implications for Roadmap

Based on this architecture research:

1. **Phase 1 (Foundation):** Model loading + basic inspection UI
   - Low risk, validates core integration
   - Deliverable: Working Marimo notebook that loads and displays model structure

2. **Phase 2 (Parameters):** Parameter viewing from JSON
   - Low risk, read-only operations
   - Deliverable: Interactive table of scale/zero-point values

3. **Phase 3 (Inference):** Hook-based activation capture
   - Medium risk, needs validation on quantized models
   - Deliverable: Side-by-side FP32 vs INT8 comparison

4. **Phase 4 (Editing):** Parameter modification experiments
   - Higher risk, TorchScript limitations
   - Deliverable: Interactive scale/zero-point modification with re-inference

5. **Phase 5 (Polish):** Integrated playground
   - Integration risk, but all components proven
   - Deliverable: Complete playground notebook

**Research flags:**
- Phase 3 may need deeper research on quantized model hooks
- Phase 4 may need research on TorchScript modification limitations
