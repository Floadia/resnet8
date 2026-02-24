## Overview

Add explicit device resolution for PyTorch ResNet8 evaluation with an automatic
default and user override. By default, evaluation should use CUDA when
available and otherwise run on CPU. Users can override this with `--device`.

## Decisions

- Add `--device` option to `scripts/evaluate_pytorch.py` with supported values:
  `auto` (default), `cpu`, `cuda`.
- Resolve runtime device as:
  - `auto` -> `cuda` if `torch.cuda.is_available()` else `cpu`
  - `cpu` -> always CPU
  - `cuda` -> require CUDA availability, otherwise raise a clear error
- Pass the resolved device into `PyTorchAdapter` and ensure both model and
  batch tensors are consistently moved to that device.
- Preserve evaluation report schema and metric computation behavior.

## Tradeoffs

- `auto` improves out-of-the-box performance on GPU hosts, but may hide device
  differences unless users pin `--device`.
- Failing fast on `--device cuda` without CUDA is stricter than silent fallback,
  but avoids ambiguous performance and correctness expectations.
