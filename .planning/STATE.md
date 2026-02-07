# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Accurate model conversion across frameworks -- converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.4 Quantization Playground - Phase 15 (Parameter Inspection)

## Current Position

Phase: 15 of 17 (Parameter Inspection)
Plan: 01 of 02 (completed)
Status: In progress
Last activity: 2026-02-07 -- Completed 15-01-PLAN.md

Progress: [================....] 18/20 plans (90% completion)

## Performance Metrics

**Velocity:**
- Total plans completed: 18
- Average duration: ~1h 37min
- Total execution time: ~29 hours 3min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| v1.0 (1-2) | 2 | ~1h | ~30min |
| v1.1 (3-4) | 2 | ~1h | ~30min |
| v1.2 (5-8) | 4 | ~2h | ~30min |
| v1.3 (9-13) | 7 | ~3.5h | ~30min |
| v1.4 (14+) | 3 | ~21.5h | ~7h 10min |

**Recent Trend:**
- Last 5 plans: 12-02, 13-01, 14-01, 14-02, 15-01
- Trend: Phase 15-01 completed in 3min (fast parameter extraction utilities)

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
- Use onnx.numpy_helper.to_array for tensor extraction (not hand-rolled raw_data parsing)
- Dequantization formula: (raw.astype(float32) - zero_point) * scale
- Per-channel quantization detected via scale.ndim > 0 (1D array vs scalar)
- Dict-based dropdown options {\"Display [Q]\": \"value\"} keeps .value clean while showing indicators

### Pending Todos

None

### Blockers/Concerns

None

## Session Continuity

Last session: 2026-02-07
Stopped at: Completed 15-01-PLAN.md (Parameter extraction utilities)
Resume file: None

**Next action:** Execute Plan 15-02 (Parameter Visualization)

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-07 with Phase 15-01 complete*
