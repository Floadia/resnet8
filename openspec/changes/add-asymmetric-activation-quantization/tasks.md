## 1. CLI Wiring

- [x] 1.1 Add `--aq-scheme` option to `scripts/evaluate_pytorch.py` and pass through to `PyTorchAdapter`.

## 2. Adapter Implementation

- [x] 2.1 Add asymmetric activation fake quantization path with zero-point handling.
- [x] 2.2 Keep weight quantization symmetric-only.
- [x] 2.3 Support calibration-derived fixed activation params for asymmetric mode.

## 3. Verification

- [x] 3.1 Add adapter unit tests for asymmetric quantization math and calibrated params.
- [x] 3.2 Run lint and targeted tests.
