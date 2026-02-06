# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Accurate model conversion across frameworks -- converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.4 Quantization Playground - Phase 15 (Parameter Inspection)

## Current Position

Phase: 15 of 17 (Parameter Inspection)
Plan: Ready to plan
Status: Ready for Phase 15
Last activity: 2026-02-06 -- Phase 14 complete (verified)

Progress: [===============.....] 17/20 plans (85% completion)

## Performance Metrics

**Velocity:**
- Total plans completed: 17
- Average duration: ~1h 42min
- Total execution time: ~29 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| v1.0 (1-2) | 2 | ~1h | ~30min |
| v1.1 (3-4) | 2 | ~1h | ~30min |
| v1.2 (5-8) | 4 | ~2h | ~30min |
| v1.3 (9-13) | 7 | ~3.5h | ~30min |
| v1.4 (14+) | 2 | ~21.5h | ~10h 45min |

**Recent Trend:**
- Last 5 plans: 12-01, 12-02, 13-01, 14-01, 14-02
- Trend: Phase 14-02 required debugging across sessions (path/import issues)

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

### Pending Todos

None

### Blockers/Concerns

None

## Session Continuity

Last session: 2026-02-06
Stopped at: Phase 14 complete (Notebook Foundation verified)
Resume file: None

**Next action:** Run `/gsd:discuss-phase 15` or `/gsd:plan-phase 15` for Parameter Inspection

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-06 with Phase 14 complete*
