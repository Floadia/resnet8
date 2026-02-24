## 1. CLI and Device Resolution

- [ ] 1.1 Add `--device {auto,cpu,cuda}` to `scripts/evaluate_pytorch.py` with
  default `auto`.
- [ ] 1.2 Implement device resolution logic:
  `auto -> cuda if available else cpu`; explicit `cuda` must error if
  unavailable.

## 2. Adapter Wiring

- [ ] 2.1 Pass the resolved device into `PyTorchAdapter`.
- [ ] 2.2 Ensure model parameters and inference tensors are moved to the
  resolved device consistently.

## 3. Verification

- [ ] 3.1 Add tests for `auto`, explicit `cpu`, and explicit `cuda` behavior on
  CUDA-available/unavailable paths.
- [ ] 3.2 Run targeted evaluation tests and confirm report schema/metrics remain
  unchanged.
