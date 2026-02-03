# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** All milestones complete (v1.0-v1.3)

## Current Position

Phase: 12 of 12 (Architecture Documentation)
Plan: 2 of 2
Status: All phases complete
Last activity: 2026-02-03 — Phase 13 removed, milestone v1.3 complete

Progress: [████████████] 100% (12/12 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 14
- Average duration: 3.93min (v1.2+ tracking)
- Total execution time: 55min (v1.2-v1.3)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Model Conversion | 1/1 | - | - |
| 2. Accuracy Evaluation | 1/1 | - | - |
| 3. PyTorch Conversion | 1/1 | - | - |
| 4. PyTorch Evaluation | 1/1 | - | - |
| 5. Calibration Infrastructure | 1/1 | 2min | 2min |
| 6. ONNX Runtime Quantization | 1/1 | 6min | 6min |
| 7. PyTorch Quantization | 1/1 | 6min | 6min |
| 8. Comparison Analysis | 1/1 | 3min | 3min |
| 9. Operation Extraction Scripts | 1/1 | 2min | 2min |
| 10. Boundary Operations Documentation | 1/1 | 2min | 2min |
| 11. Core Operations Documentation | 2/2 | 9min | 4.5min |
| 12. Architecture Documentation | 2/2 | 9min | 4.5min |

**Recent Trend:**
- Last 3 plans: 11-02 (4min), 12-01 (6min), 12-02 (3min)
- Documentation phases maintaining excellent pace (cross-referencing pattern established)

## Accumulated Context

### Decisions

Recent decisions from PROJECT.md Key Decisions table:
- ONNX as intermediate format: Standard interchange format (Good)
- tf2onnx for Keras→ONNX: Standard tool, well-maintained (Good)
- onnx2torch for ONNX→PyTorch: Leverage existing ONNX model (Good - works with FX mode)
- Separate converter/eval scripts: Reusability and clarity (Good)
- Raw pixel values (0-255): Match Keras training preprocessing (Good)

From v1.2 (PTQ Evaluation):
- ONNX Runtime uint8 recommended for best accuracy retention (86.75%, -0.44% drop)
- All quantized models meet >85% accuracy and <5% drop requirements
- PyTorch uint8 documented as not supported (fbgemm limitation)
- Per-channel quantization disabled for initial validation (per-tensor used)

From v1.3 roadmap creation:
- 5 phases derived from requirements (9-13): extraction scripts, boundary ops, core ops, architecture, hardware guide
- Research suggests minimal stack additions: only pydot + graphviz for visualization
- GitHub MathJax support validated for LaTeX math in documentation

From Phase 9 (Operation Extraction Scripts):
- Use onnx.helper.get_attribute_value() instead of direct .f/.i/.s access for proper union type handling
- Build initializer lookup dict once at start to avoid repeated iteration (O(N) not O(N*M))
- Convert numpy types to Python types for JSON serialization (float()/int() for scalars, tolist() for arrays)
- Use subprocess.run() for dot command instead of deprecated pydot.write_png()

From Phase 10 (Boundary Operations Documentation):
- Use exact ONNX specification variable names (y_scale, y_zero_point, x_scale, x_zero_point) for consistency
- Document symmetric (zero_point=0) and asymmetric quantization cases separately for clarity
- GitHub markdown math syntax: $$...$$ for display, $...$ for inline, escape underscores (y\_scale)
- Round-trip error bound: |x - Dequant(Quant(x))| ≤ y_scale/2 for values within quantization range
- Hardware pseudocode deferred for boundary operations per user decision (focusing on integer matmul instead)

From Phase 11 (Core Operations Documentation):
- QDQ format (QuantizeLinear/DequantizeLinear pairs) used in actual models instead of QLinearConv operations
- Document operations based on ONNX specification when model format differs - spec is authoritative source
- INT32 accumulator is non-negotiable: 64 channels × 3×3 kernel = 9.3M accumulator (283.5× INT16 max)
- Per-channel quantization overhead is negligible for typical layers (0.17% for 256-channel conv)
- Two-stage computation pattern (INT8×INT8→INT32 MAC, then requantization) is universal across quantized ops
- Validation scripts with multiple test cases demonstrate correctness and edge case handling
- Cross-referencing pattern: Link to detailed explanations in related operation docs instead of duplicating content
- QLinearMatMul shares identical arithmetic pattern with QLinearConv (only structural differences: no spatial dims, different input names)

From Phase 12 (Architecture Documentation):
- QDQ format operations process FP32 data, not INT8 (critical distinction: INT8 storage, FP32 computation)
- ONNX Runtime fuses Q-DQ-Op patterns into INT8 kernels at inference time for performance
- ResNet8 has 32 QuantizeLinear + 66 DequantizeLinear = 98 QDQ nodes (75% of graph)
- Asymmetry in Q vs DQ counts due to pre-quantized weights stored as initializers
- JSON-driven visualization more portable than ONNX-driven (no library dependencies)
- Conceptual diagrams more effective than full graph for understanding (130 nodes overwhelming)
- Scale/zero-point parameters stored as initializers with systematic naming convention
- Residual connections have significant scale mismatches (2.65×-3.32× ratios in ResNet8)
- QDQ dequant-add-quant pattern required for mathematically correct residual addition
- Direct INT8 addition fails when branches have different scales (same value represents different magnitudes)
- PyTorch→ONNX conversion: export FP32 then quantize with ONNX Runtime (avoids aten::quantize_per_channel limitation)
- Two-stage computation pattern applies to both QLinear operators (spec) and QDQ format (implementation)

### Pending Todos

None - all milestones complete

### Blockers/Concerns

**No blockers.** All planned work complete.

## v1.3 Milestone Overview

**Goal:** Create reference documentation explaining quantized inference calculations for hardware implementation

**Phase structure:**
- Phase 9: Operation extraction/visualization tools (enables data-driven documentation)
- Phase 10: Boundary operations (QuantizeLinear/DequantizeLinear)
- Phase 11: Core operations (QLinearConv/QLinearMatMul)
- Phase 12: Architecture (data flow, residual connections, PyTorch equivalents)

**Status:** COMPLETE (Phase 13 removed - hardware guide deferred)

## Session Continuity

Last session: 2026-02-03
Stopped at: All milestones complete
Resume file: None

**Next action:** Start new milestone with `/gsd:new-milestone` or review completed work

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-03 with Phase 13 removed, v1.3 milestone complete*
