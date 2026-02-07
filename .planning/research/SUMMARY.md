# Project Research Summary

**Project:** ResNet8 v1.4 Quantization Playground
**Domain:** Interactive ML experimentation tool (Marimo notebook)
**Researched:** 2026-02-05
**Confidence:** HIGH

## Executive Summary

The Quantization Playground milestone extends the existing ResNet8 project with an interactive Marimo notebook that allows users to inspect and experiment with quantization parameters. The project already has strong foundations: quantized models exist (ONNX and PyTorch), parameter extraction scripts work (`extract_operations.py`), and evaluation infrastructure is in place. The playground adds interactivity and comparison capabilities on top of these existing components.

The recommended approach is to build the playground as a **thin wrapper layer** over existing scripts, not as a reimplementation. Marimo notebooks will import from `scripts/` and add reactive UI controls. The stack additions are minimal: just Marimo (>=0.19.7) for the notebook environment and Plotly (>=6.5.0) for interactive visualizations. No heavy dependencies like onnx-graphsurgeon are needed because existing infrastructure already extracts what's needed.

Key risks center on Marimo's reactive execution model and quantized model internals. The most critical pitfalls are: (1) model reloading on every slider change exhausting memory, (2) in-place mutations not triggering reactive updates, and (3) the complexity of capturing intermediate values and modifying quantized weights. All are preventable with proper cell isolation, immutable data patterns, and explicit run buttons for expensive operations.

## Key Findings

### Recommended Stack

The existing stack (PyTorch, ONNX Runtime, onnx, numpy) remains unchanged. Two additions are needed for the interactive notebook layer.

**Core additions:**
- **marimo** (>=0.19.7): Reactive notebook environment with built-in widgets. Pure Python files are git-friendly. Chosen over Jupyter for reactivity and version control benefits.
- **plotly** (>=6.5.0): Interactive heatmaps and charts with native Marimo integration via `mo.ui.plotly()`. Chosen over matplotlib because matplotlib plots are not reactive in Marimo.

**No changes needed:**
- onnx (>=1.17.0), onnxruntime (>=1.23.2), torch (>=2.0.0), numpy (>=1.26.4) -- all remain as-is
- No onnx-graphsurgeon, altair, tensorboard, or additional heavy dependencies

### Expected Features

**Must have (table stakes):**
- Load quantized models (ONNX and PyTorch)
- Display model structure (reuse `visualize_graph.py`, `annotate_qdq_graph.py`)
- List all layers with scale/zero-point values (reuse `extract_operations.py`)
- Select layer and view details (Marimo dropdowns)
- Run inference on CIFAR-10 sample (reuse `evaluate.py` logic)
- Display accuracy baseline

**Should have (differentiators):**
- Intermediate value capture during inference (PyTorch hooks, ONNX model modification)
- FP32 vs INT8 side-by-side comparison
- SQNR (Signal-to-Quantized-Noise Ratio) calculation per layer
- Activation histogram visualization
- Interactive scale/zero-point modification with re-inference

**Defer to v2+:**
- PyTorch model support in playground (start with ONNX only for simplicity)
- Per-layer quantization toggle
- Export modified model to disk
- Residual connection analysis

### Architecture Approach

The architecture follows a **wrapper pattern**: Marimo notebooks import and call existing utility functions from `scripts/`, display results interactively, and allow parameter modification through reactive UI elements. A new `playground/` package provides Marimo-specific utilities that bridge between the notebook layer and existing scripts.

**Major components:**
1. **Marimo Notebook Layer** -- UI elements (dropdowns, sliders, tables, plots) for interaction
2. **playground/ Utilities** -- model_loader.py, param_inspector.py, inference.py, comparison.py
3. **Existing Scripts Layer** -- calibration_utils.py, evaluate_pytorch.py, extract_operations.py (unchanged)
4. **Data Layer** -- models/ (resnet8.pt, resnet8_int8.pt, operations.json) + CIFAR-10 dataset

**Key pattern:** Refactor only where necessary to make existing scripts importable; prefer wrapping over modifying.

### Critical Pitfalls

1. **Model reloading on slider changes** -- Use `mo.cache` for expensive model loading and isolate loading cells from UI dependencies. Model reload should only happen when model path changes, never when sliders move.

2. **Object mutations not triggering updates** -- Marimo tracks variable assignments, not in-place mutations. Always create new objects instead of mutating (e.g., `modified_scales = {**original_scales, "key": new_value}`).

3. **UI element values resetting** -- Isolate UI element definitions in cells with minimal dependencies. Use `mo.state` for values that must persist across cell reruns.

4. **Intermediate value capture requires model surgery** -- ONNX Runtime doesn't expose intermediate tensors by default. Use `onnxruntime.quantization.qdq_loss_debug.modify_model_output_intermediate_tensors()` or add outputs manually to ONNX graph.

5. **Modifying quantized weights has special requirements** -- Quantized tensors use internal representation (qint8). Cannot modify via state_dict. Must use proper APIs: `torch._make_per_channel_quantized_tensor()` for PyTorch, `onnx.numpy_helper` for ONNX initializers.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Notebook Foundation
**Rationale:** Establish core infrastructure before adding features. Validates Marimo + PyTorch integration and prevents memory/reactivity issues from the start.
**Delivers:** Working Marimo notebook that loads models and displays structure using existing `extract_operations.py` output.
**Addresses:** Model loading, structure display, layer listing (table stakes)
**Avoids:** Model reloading pitfall by establishing `mo.cache` pattern early

### Phase 2: Parameter Inspection (Read-Only)
**Rationale:** Build on Phase 1 with parameter viewing. Still read-only, no modifications yet.
**Delivers:** Interactive table of scale/zero-point values per layer, layer selection dropdown, parameter visualization.
**Uses:** Existing `resnet8_int8_operations.json` via `param_inspector.py`
**Implements:** Parameter viewing component of playground architecture

### Phase 3: Inference with Hooks
**Rationale:** Intermediate value capture is prerequisite for comparison features. Needs validation that hooks work on quantized models.
**Delivers:** Ability to run inference and capture activations at any layer. FP32 vs INT8 side-by-side comparison.
**Avoids:** Intermediate capture pitfall by implementing proper model modification patterns

### Phase 4: Interactive Modification
**Rationale:** Most complex phase, depends on all previous. Requires understanding of quantized tensor internals.
**Delivers:** Sliders for scale/zero-point modification, re-run inference after changes, output comparison.
**Avoids:** UI reset pitfall via cell isolation; slider performance pitfall via `mo.ui.run_button` pattern

### Phase 5: Integrated Playground
**Rationale:** Polish and integration after all components work individually.
**Delivers:** Single notebook combining all features with good UX. Quick evaluation for interactive feedback.
**Avoids:** Evaluation speed pitfall via tiered evaluation (100-sample quick eval vs full 10K eval)

### Phase Ordering Rationale

- **Foundation first:** Memory and reactivity patterns must be established before adding complexity. Phases 1-2 are low risk and validate core integration.
- **Read before write:** Inspection (Phases 1-2) before modification (Phases 3-4). Understanding what exists before changing it.
- **Hook validation before comparison:** Phase 3 validates that hooks work on quantized models before building comparison features that depend on them.
- **Tiered evaluation:** Quick evaluation pattern introduced in Phase 5 prevents the common mistake of running full 10K-image evaluation on every parameter change.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Inference with Hooks):** PyTorch JIT-traced models may have limited hook support. Needs validation during implementation. Fallback: capture activations before JIT tracing.
- **Phase 4 (Interactive Modification):** TorchScript model modification limitations unknown. May need to load checkpoint format instead of JIT. Verify modification patterns actually change outputs.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Well-documented Marimo patterns. `mo.cache` and cell isolation are standard.
- **Phase 2 (Parameter Inspection):** Read-only, uses existing JSON extraction. Standard dataframe/table display.
- **Phase 5 (Integration):** Composition of proven components. Focus on UX polish.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | marimo and plotly versions verified via PyPI 2026-02-05. Minimal additions to proven stack. |
| Features | MEDIUM | Table stakes clear from existing tools (Netron, PyTorch Numeric Suite). Differentiators need validation. |
| Architecture | HIGH | Wrapper pattern is straightforward. Existing scripts already modular. |
| Pitfalls | HIGH | Marimo pitfalls from official docs. Quantization pitfalls from PyTorch forum and ONNX docs. |

**Overall confidence:** HIGH

### Gaps to Address

- **JIT model hook compatibility:** PyTorch JIT-traced models may not support forward hooks. Test early in Phase 3; have fallback plan.
- **Parameter modification verification:** Must verify that modifications actually affect inference output. Add assertion checks during Phase 4.
- **Large tensor visualization:** High-dimensional weights need reduction strategy for 2D display. Address in Phase 2 with mean-over-channels or PCA approach.

## Sources

### Primary (HIGH confidence)
- [marimo PyPI](https://pypi.org/project/marimo/) -- version 0.19.7 verified
- [plotly PyPI](https://pypi.org/project/plotly/) -- version 6.5.2 verified
- [Marimo Best Practices](https://docs.marimo.io/guides/best_practices/) -- mutation, caching patterns
- [Marimo Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/) -- mo.cache, mo.stop patterns
- [PyTorch Forward Hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html)
- [ONNX Runtime Quantization Debug](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### Secondary (MEDIUM confidence)
- [PyTorch Changing Quantized Weights](https://discuss.pytorch.org/t/changing-quantized-weights/109060) -- modification patterns
- [sklearn-onnx Intermediate Results](https://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html) -- ONNX intermediate capture
- [PyTorch Numeric Suite Tutorial](https://docs.pytorch.org/tutorials/prototype/numeric_suite_tutorial.html) -- SQNR calculation

### Tertiary (LOW confidence)
- JIT model hook behavior -- needs validation during implementation
- Marimo + ONNX Runtime integration -- limited community examples, derive from general patterns

---
*Research completed: 2026-02-05*
*Ready for roadmap: yes*
