## 1. CLI and Wiring

- [x] 1.1 Add `--pre-channel` flag in `scripts/evaluate_pytorch.py` and pass it to `PyTorchAdapter`.
- [x] 1.2 Update PTQ run summary text to include the selected weight quantization granularity.

## 2. Per-Channel PTQ Implementation

- [x] 2.1 Extend `PyTorchAdapter` to accept a per-channel toggle for weight PTQ.
- [x] 2.2 Implement symmetric per-channel fake quantization for model weights and preserve existing per-tensor behavior when disabled.
- [x] 2.3 Reflect per-channel/per-tensor scheme in `describe_quantization()` output.

## 3. Verification

- [x] 3.1 Add adapter-level unit tests that verify per-channel quantization path and metadata.
- [x] 3.2 Run relevant tests for `tests/test_evaluation_adapters.py`.
