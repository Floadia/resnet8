## 1. CLI and Adapter Wiring

- [x] 1.1 Add `--calib` flag to `scripts/evaluate_pytorch.py` and pass calibration mode into `PyTorchAdapter`.
- [x] 1.2 Resolve calibration source as `<data-dir>/test_batch` when calibration mode is enabled.

## 2. Calibrated PTQ Implementation

- [x] 2.1 Extend quantization utilities to support fixed scale calibration mode.
- [x] 2.2 Derive calibration parameters from CIFAR `test_batch` and apply them for weight/activation PTQ.

## 3. Verification

- [x] 3.1 Add tests for calibrated-vs-non-calibrated quantization behavior.
- [x] 3.2 Run DoD command with `--calib` and compare accuracy against the same command without `--calib`.
