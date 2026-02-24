## Overview

Implement lightweight PTQ simulation at the PyTorch adapter layer so CLI input controls quantization behavior while keeping the backend-agnostic pipeline unchanged.

## Decisions

- Add optional `weight_bits` and `activation_bits` to `PyTorchAdapter`.
- Apply weight PTQ once on model load using symmetric fake quantization.
- Apply activation PTQ via forward hooks on key compute modules (`Conv2d`, `Linear`, `OnnxMatMul`, and selected activation/pooling modules).
- Validate bit-width range in adapter (`2..16`) and surface invalid values as argument errors.

## Tradeoffs

- This is fake quantization (quantize-dequantize) for evaluation comparability, not integer-kernel execution.
- Hook-based activation quantization is minimally invasive but approximates PTQ behavior.
