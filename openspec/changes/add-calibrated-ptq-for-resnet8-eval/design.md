## Overview

Introduce an optional calibration path for PTQ simulation in the PyTorch adapter. When enabled, calibration computes fixed symmetric scales from representative CIFAR test data and applies these scales for weight and activation quantization during evaluation.

## Decisions

- Add `--calib` flag in `scripts/evaluate_pytorch.py`.
- When `--calib` is set, load calibration tensors from `<data-dir>/test_batch`.
- Extend `PyTorchAdapter` with an optional calibration mode and calibration images input.
- Use calibrated symmetric scale for weight quantization and activation hook quantization.
- Preserve legacy dynamic-per-tensor fake quantization when calibration mode is disabled.

## Tradeoffs

- Using `test_batch` as calibration data is simple and deterministic for this workflow, but mixes calibration and evaluation distribution.
- Hook-based activation quantization remains an approximation of deployment integer kernels.
