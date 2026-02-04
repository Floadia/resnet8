# Phase 11: Core Operations Documentation — Context

## Overview

Document QLinearConv and QLinearMatMul operations for someone implementing CNN on analog accelerator, using ResNet8 as the reference model.

## Decisions

### Explanation Depth

- **Target audience:** Analog accelerator implementer learning quantization fundamentals
- **Prerequisites assumed:** Convolution basics, ONNX familiarity — no need to explain these
- **Focus:** Practical INT8 operations, not mathematical rigor
- **Edge cases:** Cover overflow handling, saturation, rounding modes — full picture
- **Formula style:** Show formulas with INT8 emphasis, step-by-step practical flow

### Worked Examples

- **Coverage:** Every layer in ResNet8 — complete coverage
- **Spatial dimensions:** Show method once, note it applies to all sizes (32×32, 16×16, etc.)
- **Numeric values:** Use actual INT8 values from the exported ONNX model
- **Hardware detail:** Skip hardware-specific accumulator tracking — focus on quantization math

### Pseudocode Style

- **Format:** Runnable Python/PyTorch with visualization
- **Quantization code:** Raw arithmetic (`(x / scale).round().clamp(-128, 127)`) — not high-level torch.quantization APIs
- **Convolution explanation:** Skip — only document what quantization changes from standard convolution
- **Verification:** Include snippets that load ONNX values and confirm calculations match expected output

### Per-channel vs Per-tensor

- **Structure:** Per-tensor first (simpler case), then per-channel as extension
- **ResNet8 mapping:** Document which layers use which approach
- **Explanation depth:** Both "how" (scale arrays, arithmetic changes) and "why" (varying weight ranges)
- **Practical implications:** Include — storage requirements, requantization differences

## Deferred Ideas

(None captured during discussion)

## Next Steps

- `/gsd:plan-phase 11` — Create execution plan
- `/gsd:research-phase 11` — Research implementation details first
