# Feature Landscape: Quantization Playground

**Domain:** Interactive quantization inspection/experimentation tool
**Researched:** 2026-02-05
**Milestone:** v1.4 Quantization Playground
**Confidence:** MEDIUM (verified against existing tools and community patterns)

## Executive Summary

A quantization playground for ResNet8 serves users who want to "see every factor" in their quantized models. Based on research of existing tools (Netron, PyTorch Numeric Suite, NVIDIA Nsight, MATLAB Deep Network Quantizer) and the project's existing infrastructure (extract_operations.py, annotate_qdq_graph.py), the feature set falls into three categories: inspection (view parameters), capture (intermediate values), and experimentation (modify and compare).

The project already has strong foundations for parameter extraction. The playground adds interactivity and comparison capabilities via Marimo's reactive notebook environment.

---

## Table Stakes

Features users expect. Missing = tool feels incomplete.

| Feature | Why Expected | Complexity | Existing Foundation | Notes |
|---------|--------------|------------|---------------------|-------|
| **Load quantized models** | Entry point for any inspection | Low | `quantize_onnx.py`, `quantize_pytorch.py` | Both ONNX and PyTorch models already exist |
| **Display model structure** | Users need graph overview | Low | `visualize_graph.py`, `annotate_qdq_graph.py` | Existing DOT/PNG generation can be embedded |
| **List all layers with types** | Basic navigation requirement | Low | `extract_operations.py` | JSON extraction already implemented |
| **Show scale/zero-point per layer** | Core quantization params | Low | `extract_operations.py` | Already extracts scales/zp from ONNX |
| **Run inference on sample input** | Verify model works | Low | `evaluate.py`, `evaluate_pytorch.py` | Evaluation scripts exist |
| **Display final accuracy** | Baseline metric users expect | Low | `evaluate.py` | Returns accuracy percentage |
| **Select specific layer to inspect** | Navigate large graphs | Low | None (new UI) | Marimo dropdowns/selectors |
| **Show tensor shapes at each layer** | Debug dimension mismatches | Low | ONNX shape inference | Standard ONNX API |

**Table Stakes Summary:** Basic inspection of what's already extracted. The existing scripts provide 80% of the data; the playground surfaces it interactively.

---

## Differentiators

Features that set product apart. Not expected, but add real value for understanding quantization.

### Tier 1: High Value, Moderate Complexity

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Intermediate value capture** | See actual tensor values at any layer during inference | Medium | PyTorch hooks, ONNX intermediate outputs | Users want to "see every factor" |
| **FP32 vs INT8 comparison** | Understand quantization impact per layer | Medium | Original + quantized models | Side-by-side value comparison |
| **SQNR (Signal-to-Quantized-Noise Ratio)** | Quantify layer sensitivity | Medium | FP32 reference values | Industry-standard metric from PyTorch Numeric Suite |
| **Activation histogram** | Visualize value distribution | Medium | Captured intermediate values | Shows if clipping is appropriate |
| **Scale factor visualization** | Graph scale values across layers | Low | Existing extraction | Helps spot outliers |

### Tier 2: Advanced Experimentation

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Modify scale interactively** | Experiment with different quantization parameters | Medium-High | Model graph manipulation | Marimo sliders drive re-quantization |
| **Modify zero-point interactively** | Fine-tune asymmetric quantization | Medium-High | Model graph manipulation | Need to validate value ranges |
| **Re-run inference after modification** | See impact of changes | Medium | Modified model creation | Marimo reactivity handles dependency |
| **Compare original vs modified outputs** | A/B testing of parameters | Medium | Dual inference paths | Output diff visualization |
| **Per-layer quantization toggle** | Enable/disable quantization per layer | High | Selective dequantization | Identify problematic layers |

### Tier 3: Advanced Analysis

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Residual connection analysis** | ResNet-specific: scale mismatch at Add nodes | Medium | Graph traversal | STATE.md notes 2.65x-3.32x scale ratios |
| **Weight distribution histogram** | See weight value spread per layer | Low | Weight tensor access | Standard visualization |
| **Sensitivity ranking** | Sort layers by quantization impact | Medium | SQNR per layer | Guide optimization efforts |
| **Export modified model** | Save experimental changes | Medium | ONNX model writing | For deployment testing |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full retraining interface** | Out of scope (PTQ only per PROJECT.md) | Link to external QAT resources |
| **Real-time inference benchmarking** | Performance not in scope per PROJECT.md | Focus on accuracy, not speed |
| **Custom dataset upload** | CIFAR-10 only per constraints | Pre-load CIFAR-10 samples |
| **Automatic optimization suggestions** | Requires expertise to validate | Show data, let user decide |
| **GPU inference support** | Adds complexity, CPU sufficient for ResNet8 | Stick with CPU inference |
| **TFLite/other format support** | Out of scope per PROJECT.md | ONNX and PyTorch only |
| **Multi-model comparison** | Scope creep | Single model focus per session |
| **Cloud deployment** | Local notebook sufficient | Marimo runs locally |
| **Real-time video/stream input** | CIFAR-10 is image classification | Single image inference |
| **Undo/redo for modifications** | Notebook handles state via re-run | Reactive cells are the "undo" |

---

## Feature Dependencies

```
Load Model (table stakes)
    |
    +---> Display Structure (table stakes)
    |         |
    |         +---> Select Layer (table stakes)
    |                   |
    |                   +---> Show Parameters (table stakes)
    |                   |
    |                   +---> Intermediate Capture (differentiator)
    |                             |
    |                             +---> Activation Histogram (differentiator)
    |                             |
    |                             +---> FP32 Comparison (differentiator)
    |                                       |
    |                                       +---> SQNR Calculation (differentiator)
    |
    +---> Run Inference (table stakes)
              |
              +---> Display Accuracy (table stakes)
              |
              +---> Modify Parameters (differentiator)
                        |
                        +---> Re-run Inference (differentiator)
                                  |
                                  +---> Compare Outputs (differentiator)
```

**Critical Path:** Load Model -> Display Structure -> Select Layer -> Show Parameters
**Experimentation Path:** Requires intermediate capture first, then comparison

---

## MVP Recommendation

For MVP (first working version), prioritize:

### Must Have (Phase 1)
1. **Load ONNX quantized model** - Entry point
2. **Display model structure** - Use existing annotate_qdq_graph output
3. **List layers with scale/zero-point** - Use existing extract_operations output
4. **Select layer and view details** - Marimo dropdown + reactive display
5. **Run inference on CIFAR-10 sample** - Use existing evaluate.py logic
6. **Display accuracy** - Single number baseline

### Should Have (Phase 2)
1. **Intermediate value capture** - Enables the "see every factor" goal
2. **FP32 vs INT8 side-by-side** - Core comparison use case
3. **Activation histogram per layer** - Visual understanding
4. **SQNR calculation** - Quantitative comparison

### Could Have (Phase 3)
1. **Interactive scale modification** - Experimentation
2. **Re-run inference after changes** - See impact
3. **Output comparison visualization** - Diff display

### Defer to Post-v1.4
- **PyTorch model support** - Start with ONNX, add PyTorch later
- **Per-layer quantization toggle** - Complex graph manipulation
- **Export modified model** - Validation and format handling
- **Residual connection analysis** - Advanced/specialized

---

## Marimo-Specific Considerations

Marimo's reactive cell model affects feature implementation:

| Marimo Feature | How It Helps | Feature It Enables |
|----------------|--------------|-------------------|
| **Reactive cells** | Change input, outputs update automatically | Interactive parameter modification |
| **UI elements (sliders, dropdowns)** | Native interactive controls | Layer selection, parameter adjustment |
| **Dataframe display** | Built-in table rendering | Layer listing with parameters |
| **Matplotlib/Plotly support** | Visualization integration | Histograms, comparison charts |
| **Pure Python storage** | Version control friendly | Can commit notebook to repo |
| **Script execution mode** | Can run without UI | CI/validation of notebook |

**Implication:** Marimo's reactivity means modifying a scale slider automatically triggers:
1. Model update cell
2. Inference cell
3. Comparison cell
4. Visualization cell

This is the "interactive modification and re-evaluation" from PROJECT.md requirements.

---

## Complexity Assessment by Feature Category

| Category | Total Features | Low | Medium | High |
|----------|----------------|-----|--------|------|
| Table Stakes | 8 | 8 | 0 | 0 |
| Tier 1 Differentiators | 5 | 1 | 4 | 0 |
| Tier 2 Differentiators | 5 | 0 | 3 | 2 |
| Tier 3 Differentiators | 4 | 1 | 3 | 0 |

**Observation:** Table stakes are all low complexity because existing scripts provide the data. Differentiators require new logic for capture, comparison, and modification.

---

## Existing Code Leverage

| Existing Script | Features It Enables |
|-----------------|---------------------|
| `extract_operations.py` | Layer listing, scale/zero-point display, model structure |
| `annotate_qdq_graph.py` | Visual graph display, QDQ architecture diagram |
| `evaluate.py` | Inference execution, accuracy calculation |
| `evaluate_pytorch.py` | PyTorch inference path (future) |
| `calibration_utils.py` | CIFAR-10 sample loading |
| `visualize_graph.py` | Model structure visualization |

**Recommendation:** Build playground as orchestration layer over existing scripts, not replacement. Import functions from scripts into Marimo notebook cells.

---

## Implementation Approaches

### Intermediate Value Capture

**ONNX Approach:**
```python
# Use onnxruntime intermediate outputs
# Set up session to output at desired nodes
sess = onnxruntime.InferenceSession(model_path)
output_names = [node.name for node in model.graph.node if filter_condition]
results = sess.run(output_names, {input_name: input_data})
```

**PyTorch Approach:**
```python
# Use forward hooks to capture activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for name, layer in model.named_modules():
    layer.register_forward_hook(get_activation(name))
```

### FP32 vs INT8 Comparison

**Workflow:**
1. Load both FP32 and INT8 models
2. Run same input through both
3. Capture intermediate activations at corresponding layers
4. Compute SQNR: `10 * log10(signal_power / noise_power)`
5. Display per-layer comparison table

**PyTorch Numeric Suite Reference:**
- Use `torch.ao.ns.fx.utils.compute_sqnr()` if available
- Or compute manually: `SQNR = 10 * log10(sum(fp32^2) / sum((fp32 - quant)^2))`

### Interactive Parameter Modification

**ONNX Graph Manipulation:**
```python
# Modify scale/zero-point in initializers
for init in model.graph.initializer:
    if init.name == target_scale_name:
        # Create new initializer with modified value
        new_init = numpy_helper.from_array(new_scale_value, init.name)
        # Replace in graph
```

**Marimo Integration:**
```python
@app.cell
def scale_slider():
    return mo.ui.slider(0.001, 0.1, value=0.01, label="Scale")

@app.cell  # Reactive: re-runs when slider changes
def modified_inference(scale_slider):
    modified_model = modify_scale(model, scale_slider.value)
    return run_inference(modified_model, input_data)
```

---

## Sources

- [PyTorch Numeric Suite Tutorial](https://docs.pytorch.org/tutorials/prototype/numeric_suite_tutorial.html) - SQNR and comparison methodology (HIGH confidence)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Scale/zero-point parameter details (HIGH confidence)
- [Netron GitHub](https://github.com/lutzroeder/netron) - Model visualization patterns (HIGH confidence)
- [Marimo Documentation](https://docs.marimo.io/) - Reactive notebook capabilities (HIGH confidence)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Histogram calibration methods (MEDIUM confidence)
- [NVIDIA Nsight Deep Learning Designer](https://docs.nvidia.com/nsight-dl-designer/UserGuide/index.html) - Weight editor patterns (MEDIUM confidence)
- [PyTorch Practical Quantization](https://pytorch.org/blog/quantization-in-practice/) - Debugging workflow (HIGH confidence)
- [TorchLens Paper](https://www.nature.com/articles/s41598-023-40807-0) - Intermediate activation extraction (MEDIUM confidence)
- [MATLAB Deep Network Quantizer](https://www.mathworks.com/help/deeplearning/ug/quantization-of-deep-neural-networks.html) - Quantization workflow visualization (MEDIUM confidence)

---

*Feature landscape research complete. Ready for requirements definition.*
