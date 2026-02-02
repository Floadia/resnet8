# Project Research Summary

**Project:** ResNet8 Quantized Operations Documentation
**Domain:** Neural network quantization reference documentation
**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

This milestone creates reference documentation explaining the mathematical operations needed to implement quantized neural network inference in hardware accelerators. The focus is on ONNX quantized operators (QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear) and their hardware implementation requirements. Hardware implementers need precise specifications of integer arithmetic, scale/zero-point parameters, and data flow through quantized networks.

The recommended approach combines programmatic ONNX model inspection with graph visualization and GitHub-flavored Markdown documentation. The existing stack (onnx >=1.17.0, onnxruntime >=1.23.2) already includes all necessary tools for extracting operation details. Only one new dependency (pydot + system graphviz) is needed for generating model visualizations. GitHub's native MathJax support (since May 2024) enables LaTeX math equations without additional tooling.

The key risk is complexity overload: quantized operations involve multiple stages (integer MAC, requantization, scale/zero-point handling) that can overwhelm readers if not presented incrementally. Mitigation: structure documentation layer-by-layer (single operation → residual block → full network) with concrete ResNet8 examples at each level. Critical pitfalls to highlight: insufficient accumulator bit-width (causes overflow), incorrect rounding modes (degrades accuracy 2-5%), and per-channel quantization implementation complexity.

## Key Findings

### Recommended Stack

**Minimal additions to existing stack.** The project already has ONNX Python API (onnx >=1.17.0) for model inspection and ONNX Runtime (>=1.23.2) for validation. For v1.3, only pydot (Python package) and graphviz (system package) are needed for generating graph visualizations.

**Core technologies:**
- **ONNX Python API** (already available): Programmatic model loading, node iteration, attribute extraction — official ONNX library enables scripted extraction of QLinear operation parameters
- **onnx.tools.net_drawer** (built into onnx): Generate Graphviz DOT files from ONNX models — official visualization tool produces Netron-style diagrams
- **pydot + graphviz** (new for v1.3): Convert DOT to PNG/SVG — enables automated documentation workflow with embedded diagrams
- **GitHub Markdown + MathJax** (no install): LaTeX math rendering — native support since May 2024, no custom tooling needed

**Not adding:** Netron (GUI-only, no programmatic export), onnx-tool (performance profiling out of scope), TensorFlow (only needed for original Keras model conversion, already done in v1.0-v1.1).

### Expected Features

Documentation must explain four core quantized operations with hardware-implementable formulas.

**Must have (table stakes):**
- **QuantizeLinear/DequantizeLinear**: FP32 ↔ INT8 boundary operations with exact formulas: `q = saturate(round(x/scale) + zero_point)` and `x = (q - zero_point) × scale`
- **QLinearConv**: Two-stage integer convolution (8×8→32 MAC, then requantization to 8-bit) with all 9 inputs documented (x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, bias)
- **QLinearMatMul**: Integer matrix multiplication with similar two-stage structure (8×8→32 MAC, then requantization)
- **Scale/zero-point parameters**: Where they appear in ONNX graphs (initializers), how they're used in computation, storage requirements

**Should have (competitive):**
- **Per-channel vs per-tensor quantization**: Hardware implications (scale[c] per output channel vs scalar scale), memory requirements (64 scales for Conv2D(64) vs 1 scale)
- **Residual connection handling**: Three solution approaches for scale mismatch in ResNet skip connections (dequant-add-quant, scale matching via requantization, PyTorch FloatFunctional)
- **ONNX graph visualizations**: Full ResNet8 quantized graph, zoomed-in single-operation diagrams, annotated with scale/zero-point flow
- **Hardware implementation pseudocode**: C/Verilog-style code showing exact integer arithmetic and bit-widths

**Defer (v2+):**
- **Mixed-precision quantization**: INT4 weights, INT8 activations (emerging, not standard yet)
- **Dynamic quantization**: Runtime scale computation (incompatible with hardware accelerators)
- **Calibration methodology**: How to choose scale/zero-point values (covered in v1.2 PTQ milestone, not needed for v1.3 operations documentation)
- **Performance profiling**: MACs/FLOPs counting, inference speed optimization (out of scope for reference documentation)

### Architecture Approach

Quantized inference transforms compute graphs from floating-point to integer arithmetic with explicit scale/zero-point parameters. Two formats exist: **QDQ format** (QuantizeLinear-DequantizeLinear pairs around FP32 ops) and **Pure Integer format** (QLinear ops with embedded parameters). ONNX Runtime defaults to QDQ format where runtime fuses Q/DQ pairs with operators for integer execution.

**Major components:**
1. **Boundary operations** (QuantizeLinear/DequantizeLinear) — Convert between FP32 input/output and INT8 internal representation, applied at model input and before final classification head
2. **Quantized convolution/matmul** (QLinearConv/QLinearMatMul) — Two-stage integer operations: 8×8→32-bit MAC accumulation, then requantization (scale conversion) back to 8-bit output
3. **Residual connections** (Add with scale matching) — Critical architecture point where two INT8 tensors with potentially different scales must be added, requires dequantization to FP32 or scale matching via requantization
4. **Per-channel quantization parameters** (scale[c], zero_point[c]) — Each convolution output channel has independent scale/zero-point stored in ONNX initializers, hardware must index correctly

**Data flow:** FP32 input → QuantizeLinear → INT8 tensor → (QDQ-wrapped Conv ops) → INT8 activations → DequantizeLinear → FP32 output. Key insight: QDQ format appears to be FP32 in graph, but runtime fuses operations for integer execution.

### Critical Pitfalls

Top 5 hardware implementation mistakes that break accuracy or cause incorrect results:

1. **Insufficient accumulator bit-width causes overflow** — Using 16-bit accumulators for INT8 multiply-accumulate overflows during convolution. ResNet8 Conv2D(64, 3×3) requires 25 bits worst-case (576 MACs × 127² = 9,290,304). **Prevention:** Always use INT32 accumulators for INT8×INT8 operations (industry standard).

2. **Incorrect rounding mode degrades accuracy 2-5%** — Using truncation (floor) instead of round-to-nearest during requantization causes systematic bias. **Prevention:** Implement round-to-nearest-even (banker's rounding) to match ONNX Runtime behavior: `rounded = (value >= 0) ? (value + 0.5) >> frac_bits : (value - 0.5) >> frac_bits`.

3. **Scale factors stored with insufficient precision** — Using float16 or fixed-point <24 bits for scales causes quantization range mismatch, degrading accuracy 3-10%. **Prevention:** Use float32 for scale factors (4 bytes per scale, ~160 bytes total for ResNet8, negligible memory cost).

4. **Per-channel quantization complexity underestimated** — Implementing per-tensor scale for all channels loses accuracy benefit of per-channel quantization. **Prevention:** Store C_out scales (one per output channel), multiplex correct scale[c] during requantization, verify with test cases where each channel has different scale.

5. **Fused operations incorrectly implemented** — Implementing Conv-BatchNorm-ReLU as separate operations with intermediate quantization degrades accuracy 5-10%. **Prevention:** Fold BatchNorm into Conv weights/bias offline (before quantization), implement Conv-ReLU as single fused operation with one requantization step.

**Additional moderate pitfall:** Zero-point asymmetry not handled in convolution. Formula requires correction terms: `acc - zero_x × Σ(weights)` for asymmetric activations. Recommendation: use symmetric weights (zero_w = 0) to simplify, only handle asymmetric activations.

## Implications for Roadmap

Based on research, suggested documentation structure builds understanding incrementally from single operations to full network.

### Phase 1: Operation Extraction Scripts
**Rationale:** Before writing documentation, need programmatic tools to extract operation details from quantized ONNX models (resnet8_int8.onnx, resnet8_uint8.onnx from v1.2).
**Delivers:**
- `scripts/extract_operations.py` — Parse ONNX models, output JSON with all QLinear nodes, scales, zero-points, attributes
- `scripts/visualize_graph.py` — Generate PNG/SVG diagrams using onnx.tools.net_drawer + pydot
**Addresses:** Must-have features (programmatic extraction), enables data-driven documentation
**Avoids:** Manual inspection errors, ensures documentation matches actual model parameters

### Phase 2: Boundary Operations Documentation
**Rationale:** QuantizeLinear/DequantizeLinear are simplest operations (single formula each), establish documentation style and math rendering.
**Delivers:**
- `docs/QUANTIZE_DEQUANTIZE.md` — QuantizeLinear and DequantizeLinear formulas, numerical examples, hardware pseudocode
- GitHub Markdown with LaTeX math equations (test rendering)
**Uses:** GitHub MathJax support for inline ($...$) and block ($$...$$) equations
**Implements:** Boundary operations documentation (table stakes feature)

### Phase 3: QLinearConv Documentation
**Rationale:** QLinearConv is the most complex operation (9 inputs, two-stage computation), most critical for hardware implementers.
**Delivers:**
- `docs/QLINEAR_CONV.md` — Detailed QLinearConv breakdown: integer MAC stage, requantization stage, per-channel quantization handling, hardware implementation pseudocode with exact bit-widths
- Worked example with ResNet8 Conv2D(16, 3×3) layer showing all intermediate values
**Addresses:** QLinearConv table stakes, per-channel quantization (should-have)
**Avoids:** Pitfall #1 (accumulator overflow) by specifying INT32 requirement, Pitfall #3 (scale precision) by documenting float32 storage

### Phase 4: Residual Connection Handling
**Rationale:** ResNet8's skip connections are unique architecture challenge where scales must match at Add operations.
**Delivers:**
- `docs/RESIDUAL_QUANTIZATION.md` — Scale mismatch problem explanation, three solution approaches (QDQ dequant-add-quant, TensorRT scale matching, PyTorch FloatFunctional), hardware trade-offs
- Annotated ONNX graph showing both main path and skip path with scale annotations
**Addresses:** Residual connection handling (should-have differentiator)
**Implements:** Architecture component #3 (Add with scale matching)

### Phase 5: Complete Reference Documentation
**Rationale:** Tie all operations together with full-network context and hardware implementation checklist.
**Delivers:**
- `docs/QUANTIZATION_OPERATIONS.md` — Main reference combining all operations, full ResNet8 graph visualization, data flow diagrams, hardware accelerator requirements summary
- Hardware implementation checklist covering all 6 critical pitfalls
- Cross-references to ONNX operator specs
**Addresses:** Complete table stakes coverage, critical pitfalls summary
**Avoids:** All 6 critical pitfalls by providing verification checklist

### Phase Ordering Rationale

- **Scripts first:** Extraction and visualization tools enable accurate documentation, prevent manual errors
- **Boundary ops before compute ops:** QuantizeLinear/DequantizeLinear establish math notation and GitHub rendering workflow, simpler formulas validate approach
- **QLinearConv before MatMul:** Convolution is more complex (spatial dimensions, padding, strides), documenting it first means MatMul is easier by comparison
- **Residual connections after basic ops:** Requires understanding QLinearConv first, represents integration challenge rather than single-operation documentation
- **Summary last:** Complete reference synthesizes all prior docs, adds hardware implementation guidance

**Why this grouping avoids pitfalls:**
- Incremental complexity prevents overwhelming readers
- Scripts ensure documented parameters match actual models (no guesswork)
- Worked examples with ResNet8 layers provide concrete bit-width calculations (prevents accumulator overflow)
- Hardware pseudocode specifies exact rounding/saturation logic (prevents implementation errors)

### Research Flags

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (extraction scripts):** ONNX Python API is well-documented, onnx.tools.net_drawer has tutorial examples
- **Phase 2 (boundary ops):** QuantizeLinear/DequantizeLinear operators have complete ONNX spec with formulas
- **Phase 3 (QLinearConv):** Detailed operator spec exists, GitHub issues provide clarifications on two-stage computation

**Phases needing validation during implementation:**
- **Phase 4 (residual connections):** TensorRT scale matching strategy is vendor-specific, need to verify hardware feasibility during doc writing (may simplify to QDQ approach if TensorRT method too complex)
- **Phase 5 (hardware checklist):** Verification test cases should be executable, may need to create Python reference implementations for each pitfall test

**No phases need deep research** — v1.3 is documentation of existing quantized models from v1.2, all operations are already implemented in ONNX Runtime.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | ONNX Python API officially documented, pydot/graphviz well-established, GitHub MathJax verified |
| Features | HIGH | ONNX operator specs provide exact formulas, ResNet8 quantized models from v1.2 provide concrete examples |
| Architecture | HIGH | ONNX Runtime documentation explains QDQ format, TensorRT docs cover optimization strategies |
| Pitfalls | HIGH | Verified with 2025-2026 research papers, ResNet8 PTQ results (86.75% ONNX, 85.68% PyTorch) validate proper implementation |

**Overall confidence:** HIGH

Research is based on official ONNX specifications, verified with actual ResNet8 quantized models from v1.2, and cross-referenced with recent academic research on hardware quantization challenges. The only uncertainty is TensorRT-specific optimizations (scale matching strategy), which can default to simpler QDQ approach if needed.

### Gaps to Address

**Minor gap: Optimal graph visualization layout** — ResNet8 full graph may be too cluttered for single PNG. During Phase 1, test different rankdir options (TB vs LR), potentially generate subgraph visualizations (per residual block) in addition to full graph. Fallback: provide Netron link for interactive exploration.

**Minor gap: Hardware pseudocode language choice** — Should examples be C, Verilog, or Python? Recommendation: Use C-style pseudocode for algorithm clarity, add Verilog snippets for critical operations (rounding, saturation) where hardware specifics matter. Python reference implementations for test cases.

**No critical gaps** — All core operations have complete specifications, extraction tools are proven (ONNX Python API used in industry), visualization approach validated by ONNX tutorials.

## Sources

### Primary (HIGH confidence)
- [QLinearConv - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) — Complete operator specification with inputs, attributes, formula
- [QLinearMatMul - ONNX 1.20.0 documentation](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) — Matrix multiplication operator spec
- [QuantizeLinear - ONNX 1.21.0 documentation](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) — Quantization formula and rounding mode
- [DequantizeLinear - ONNX 1.21.0 documentation](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) — Dequantization formula
- [ONNX Python API Overview](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html) — Model loading and graph traversal
- [ONNX Visualizing a Model Tutorial](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md) — onnx.tools.net_drawer usage
- [GitHub Writing Mathematical Expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) — MathJax LaTeX support

### Secondary (MEDIUM confidence)
- [How is QLinearConv calculated? - ONNX Runtime Issue #11883](https://github.com/microsoft/onnxruntime/issues/11883) — Clarifies two-stage computation (ConvInteger + requantization)
- [ConvInteger vs QLinearConv - ONNX Issue #2424](https://github.com/onnx/onnx/issues/2424) — Explains relationship between integer convolution and requantization
- [Working with Quantized Types - NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html) — Residual connection optimization strategies
- [PyTorch vision quantized ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py) — FloatFunctional.add() implementation for skip connections

### Tertiary (Research papers, vendor-specific)
- [Frontiers: Quantized CNNs hardware perspective (2025)](https://www.frontiersin.org/journals/electronics/articles/10.3389/felec.2025.1469802/full) — Hardware implementation challenges, accumulator bit-width requirements
- [Speed up integer-arithmetic-only inference via bit-shifting (Nature 2025)](https://www.nature.com/articles/s41598-025-02544-4) — Fixed-point requantization optimization (27% FPS improvement)
- [Is RTN Quantization All You Need? (arXiv 2025)](https://arxiv.org/html/2505.15909v1) — Round-to-nearest accuracy validation
- [Deep learning inference optimization for IoT: Conv2D-ReLU-BN fusion (Springer 2025)](https://link.springer.com/article/10.1007/s11227-025-07107-y) — Fusion benefits (1.53× speedup)

**Aggregated from research files:**
- STACK.md: 15 high-confidence sources (official docs, PyPI verified versions)
- FEATURES.md: 12 sources (ONNX specs, PyTorch docs, recent research)
- ARCHITECTURE.md: 9 sources (ONNX Runtime, TensorRT, PyTorch implementation)
- PITFALLS.md: 20+ sources (2025-2026 research papers, official framework docs)

---
*Research completed: 2026-02-02*
*Ready for roadmap: yes*
