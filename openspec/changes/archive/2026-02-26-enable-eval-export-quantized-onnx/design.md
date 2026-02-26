## Overview

Add an ONNX export path to the PyTorch evaluation adapter so the same model
instance used for PTQ evaluation can be exported for ONNX Runtime checks.

## Decisions

### Export source model from adapter

- Add `export_onnx()` on `PyTorchAdapter` so export uses the exact configured
  eval model (weight PTQ and optional activation quantization modules).
- Keep this behavior opt-in from CLI (`--export-onnx`).

### Materialize activation quantization for export

- Current activation quantization is hook-based; hooks are not preserved by ONNX
  export.
- During export, apply graph-local wrappers around target modules that quantize
  their outputs with static fake-quant ops so ONNX graph contains quantization
  behavior.

### Shared score verification flow

- Add CLI flag (`--verify-exported-onnx`) to evaluate exported ONNX graph via
  `OnnxRuntimeAdapter` and report absolute accuracy delta versus PyTorch eval.
- Reuse shared `evaluate_dataset()` path and a configurable threshold flag
  (`--onnx-score-tol`, default `0.02`).

## Non-goals

- Do not replace existing standalone ONNX quantization scripts.
- Do not change default eval output/report schema.
