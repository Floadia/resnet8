## Overview

Extend PyTorch PTQ simulation to support activation quantization scheme selection. The adapter continues using symmetric quantization for weights, and supports symmetric/asymmetric fake quantization for activations in both dynamic and calibration-driven modes.

## Decisions

- Add `--aq-scheme` CLI option with values `symmetric` (default) and `asymmetric`.
- Keep weight quantization symmetric regardless of activation scheme.
- Represent activation quantization with explicit parameters (`scale`, `zero_point`, `qmin`, `qmax`).
- In calibration mode, collect per-module activation min/max and derive fixed quant params.

## Tradeoffs

- Asymmetric activation quantization better matches uint-style deployment behavior but increases implementation complexity.
- Hook-based fake quantization remains approximate compared to true integer-kernel execution.
