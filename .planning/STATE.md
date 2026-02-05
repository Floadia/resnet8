# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.4 Quantization Playground

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-05 — Milestone v1.4 started

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
- Residual connections have significant scale mismatches (2.65×-3.32× ratios in ResNet8)

### Pending Todos

None

### Blockers/Concerns

None

## Session Continuity

Last session: 2026-02-05
Stopped at: Starting milestone v1.4
Resume file: None

**Next action:** Complete requirements and roadmap definition

---
*State initialized: 2026-01-27*
*Last updated: 2026-02-05 with v1.4 milestone start*
