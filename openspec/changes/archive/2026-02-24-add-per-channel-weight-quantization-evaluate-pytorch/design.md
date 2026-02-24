## Overview

Add an opt-in per-channel weight PTQ mode for PyTorch evaluation, preserving current default behavior (per-tensor).

## Decisions

- Keep existing CLI defaults unchanged; per-channel mode is enabled only when `--pre-channel` is passed.
- Extend `PyTorchAdapter` with a `per_channel` boolean for weight PTQ behavior.
- For `Conv2d` and `Linear` weight tensors, quantize per output channel (`axis=0`) using channel-wise max-abs and symmetric parameters.
- Keep activation quantization behavior unchanged.
- Expose quantization summary rows with scheme labels (`symmetric-per-tensor` or `symmetric-per-channel`).

## Tradeoffs

- Per-channel PTQ improves representational fidelity for weights but adds slight implementation complexity.
- We intentionally scope per-channel mode to floating-point weight tensors with output-channel-first layout (standard `Conv2d`/`Linear`).
