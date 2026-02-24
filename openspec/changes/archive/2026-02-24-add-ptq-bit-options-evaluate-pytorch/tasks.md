## 1. CLI and Adapter Wiring

- [x] 1.1 Add `--wq` and `--aq` options to `scripts/evaluate_pytorch.py` and pass them to `PyTorchAdapter`.
- [x] 1.2 Extend `PyTorchAdapter` to accept optional PTQ bit-width parameters and validate ranges.

## 2. PTQ Simulation Implementation

- [x] 2.1 Implement symmetric fake quantization for floating-point tensors.
- [x] 2.2 Apply weight PTQ on model load and activation PTQ via forward hooks during inference.

## 3. Verification

- [x] 3.1 Run PyTorch evaluation with `--wq=8 --aq=8`, `--wq=8 --aq=6`, and `--wq=8 --aq=4`.
- [x] 3.2 Record and compare resulting accuracy values.
