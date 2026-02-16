# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Accurate model conversion across frameworks -- converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.5 Intermediate Value Visualizer

## Current Position

**Current Milestone:** v1.5 Intermediate Value Visualizer
**Phase:** 15 - Intermediate Activation Capture
**Plan:** Complete
**Status:** Phase complete
**Progress:** ██████████ 100% (1/1 phases)

Last activity: 2026-02-16 — Completed Phase 15 Plan 01

## Performance Metrics

**Velocity:**
- Total plans completed: 18
- Average duration: ~1h 36min
- Total execution time: ~29h 6min

**By Milestone:**

| Milestone | Phases | Plans | Total Time | Avg/Plan |
|-----------|--------|-------|------------|----------|
| v1.0 | 1-2 | 2 | ~1h | ~30min |
| v1.1 | 3-4 | 2 | ~1h | ~30min |
| v1.2 | 5-8 | 4 | ~2h | ~30min |
| v1.3 | 9-13 | 7 | ~3.5h | ~30min |
| v1.4 | 14-17 | 2 | ~21.5h | ~10h 45min |
| v1.5 | 15 | 1 | ~6min | ~6min |

**Recent Trend:**
- Last 5 plans: 12-02, 13-01, 14-01, 14-02, 15-01
- Trend: Phase 15-01 completed quickly (6min) - straightforward implementation

*Updated after each plan completion*

## Accumulated Context

### Decisions

From v1.2 (PTQ Evaluation):
- ONNX Runtime uint8 recommended for best accuracy retention (86.75%, -0.44% drop)
- All quantized models meet >85% accuracy and <5% drop requirements
- PyTorch uint8 documented as not supported (fbgemm limitation)
- Per-channel quantization disabled for initial validation (per-tensor used)

From v1.3 (Documentation):
- QDQ format operations process FP32 data, not INT8 (critical distinction: INT8 storage, FP32 computation)
- ONNX Runtime fuses Q-DQ-Op patterns into INT8 kernels at inference time for performance
- ResNet8 has 32 QuantizeLinear + 66 DequantizeLinear = 98 QDQ nodes (75% of graph)
- Scale/zero-point parameters stored as initializers with systematic naming convention
- Residual connections have significant scale mismatches (2.65x-3.32x ratios in ResNet8)

From v1.4 (Quantization Playground):
- Stack additions: marimo (>=0.19.7), plotly (>=6.5.0) only
- Use mo.cache for model loading to prevent memory leaks on re-run
- Avoid mutations -- return new objects for reactive updates
- Wrapper pattern: Marimo notebook calls existing scripts/, not reimplementation
- weights_only=False needed for PyTorch quantized models (full object deserialization)
- Graceful missing file handling: return None instead of raising errors
- Extract ONNX layers from both graph.node (operations) and graph.initializer (parameters)
- Filter out PyTorch root module (empty name from named_modules) for clean layer lists
- File selection mode (not directory mode) more reliable for Marimo file browser
- PyTorch quantized models saved as dict {'model': ..., 'epoch': ...} - extract 'model' key

From v1.5 (Intermediate Value Visualizer):
- Reference implementation exists in scripts/get_resnet8_intermediate.py
- Forward hooks pattern: register on target module, capture output in closure, remove after inference
- Input normalization: torch.from_numpy(sample).to(device) -- raw pixels (0-255), no preprocessing
- Layer discovery: model.named_modules() filters empty names (root module)
- Activation capture supports complex outputs (tuples/dicts) via recursive flattening
- Always-defined pattern for Marimo cells: use if/else with default values instead of mo.stop() when return variables must exist for downstream cells
- Data routing pattern: display_entry resolver cell centralizes view mode logic for multiple consumer cells
- Color coding: orange for activations, blue for weights (visual distinction in histograms)

### Pending Todos

None

### Blockers/Concerns

None

## Session Continuity

Last session: 2026-02-16
Stopped at: Completed Phase 15-01 (Intermediate Activation Capture)
Resume file: .planning/phases/15-intermediate-activation-capture/15-01-SUMMARY.md

**Next action:** v1.5 milestone complete - see ROADMAP.md for next milestone

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-16 with Phase 15-01 completion*
