Combination table (`accuracy %`) for `wq=4..8`, `aq=4..8`:

| wq \ aq | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|
| 4 | 11.69 | 33.70 | 48.03 | 51.18 | 51.64 |
| 5 | 10.48 | 44.52 | 60.10 | 72.05 | 75.77 |
| 6 | 12.06 | 52.31 | 76.57 | 83.02 | 84.81 |
| 7 | 12.35 | 54.02 | 75.49 | 85.34 | 86.20 |
| 8 | 12.65 | 55.25 | 76.86 | 86.15 | 87.05 |

## Per-channel vs Per-tensor (Weight PTQ)

`--pre-channel` (alias: `--per-channel`) applies to `wq` (weights) only.
Activation quantization (`aq`) does not use per-channel mode in `evaluate_pytorch.py`.

Comparison on CIFAR-10 test set (`10000` samples) with `aq` unset:

| setting | accuracy % | correct/total |
|---|---:|---:|
| `wq=5` per-tensor | 75.29 | 7529/10000 |
| `wq=5` per-channel | 82.64 | 8264/10000 |
| `wq=6` per-tensor | 85.08 | 8508/10000 |
| `wq=6` per-channel | 85.70 | 8570/10000 |

Observed delta (per-channel - per-tensor):

- `wq=5`: `+7.35` pt
- `wq=6`: `+0.62` pt

Reference reports:

- `logs/acc_w5_per_tensor.json`
- `logs/acc_w5_per_channel.json`
- `logs/acc_w6_per_tensor.json`
- `logs/acc_w6_per_channel.json`
