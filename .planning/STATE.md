# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.3 Quantized Operations Documentation

## Current Position

Phase: 9 of 13 (Operation Extraction Scripts)
Plan: 1 of 1
Status: Phase complete
Last activity: 2026-02-02 — Completed 09-01-PLAN.md

Progress: [█████████░░░░] 69% (9/13 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 3.89min (v1.2+ tracking)
- Total execution time: 35min (v1.2-v1.3)

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

**Recent Trend:**
- Last plan: 09-01 (2min)
- Tooling phases very fast (minimal complexity, clear patterns)

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

### Pending Todos

Install pydot dependency before Phase 10:
- Required for visualize_graph.py to generate model visualizations
- Install via: `sudo apt-get install python3-pydot` or `pip install pydot` in virtual environment

### Blockers/Concerns

**No critical blockers.**

Minor considerations for upcoming phases:
- Graph visualization layout: ResNet8 full graph may be cluttered, may need subgraph views per residual block
- Hardware pseudocode language: Use C-style for algorithm clarity, add Verilog snippets for hardware-specific operations
- pydot installation: Required before running visualize_graph.py in Phase 10+ (not available in current environment)

## v1.3 Milestone Overview

**Goal:** Create reference documentation explaining quantized inference calculations for hardware implementation

**Phase structure:**
- Phase 9: Operation extraction/visualization tools (enables data-driven documentation)
- Phase 10: Boundary operations (QuantizeLinear/DequantizeLinear)
- Phase 11: Core operations (QLinearConv/QLinearMatMul)
- Phase 12: Architecture (data flow, residual connections, PyTorch equivalents)
- Phase 13: Hardware implementation guide (critical pitfalls, pseudocode, test vectors)

**Research highlights:**
- All operations have complete ONNX specifications
- Quantized models from v1.2 provide concrete test cases
- Critical pitfalls: INT32 accumulator overflow, round-to-nearest-even, float32 scales, per-channel indexing

## Session Continuity

Last session: 2026-02-02
Stopped at: Completed 09-01-PLAN.md
Resume file: None

**Next action:** `/gsd:plan-phase 10` to create execution plan for Boundary Operations Documentation

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-02 with Phase 9 completion*
