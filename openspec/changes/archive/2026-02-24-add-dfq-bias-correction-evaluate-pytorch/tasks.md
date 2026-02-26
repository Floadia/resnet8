## 1. CLI and Adapter Contract

- [x] 1.1 Add `--dfq-bias-corr` to `scripts/evaluate_pytorch.py` and wire it to
      `PyTorchAdapter`.
- [x] 1.2 Enforce CLI contract that `--dfq-bias-corr` requires `--wq`.
- [x] 1.3 Ensure calibration data loading from `<data-dir>/test_batch` is
      enabled for DFQ mode.

## 2. DFQ Empirical Bias Correction Implementation

- [x] 2.1 Extend `PyTorchAdapter` with a `weight_bias_correction` option and
      validation.
- [x] 2.2 Implement empirical sequential layer-wise bias correction after weight
      PTQ using calibration images.
- [x] 2.3 Surface DFQ-corrected layer metadata in quantization summaries.

## 3. Validation and Comparison

- [x] 3.1 Add adapter tests for DFQ option validation and correction behavior.
- [x] 3.2 Run test suite and lint checks for updated files.
- [x] 3.3 Run `wq=5` A/B comparison with and without DFQ option and save JSON
      reports under `logs/`.
