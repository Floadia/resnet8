# Refactor Smoke Baseline (2026-02-23)

## Evaluation smoke checks

### FP32 PyTorch

- Command:
  - `uv run python scripts/evaluate_pytorch.py --model models/resnet8.pt --data-dir /mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py --max-samples 512 --output-json logs/baseline_eval_pytorch_fp32.json`
- Result:
  - overall accuracy: `457/512 = 89.26%`
  - report: `logs/baseline_eval_pytorch_fp32.json`

### INT8 PyTorch

- Command:
  - `uv run python scripts/evaluate_pytorch.py --model models/resnet8_int8.pt --data-dir /mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py --max-samples 512 --output-json logs/baseline_eval_pytorch_int8.json`
- Result:
  - overall accuracy: `448/512 = 87.50%`
  - report: `logs/baseline_eval_pytorch_int8.json`

### ONNX runtime

- Status: skipped in this smoke run (ONNX model artifacts missing in `models/`).

## Weight visualization extraction smoke check

- Command:
  - `uv run python` script using `playground.utils.tensor_extractors.load_tensor_index`
- Summary report:
  - `logs/baseline_weight_visualizer_summary.json`
- Key output:
  - `models/resnet8.pt`: layers `9`, tensors `18`, quantized tensors `0`
  - `models/resnet8_int8.pt`: layers `9`, tensors `18`, quantized tensors `9`
  - ONNX models: missing (`resnet8.onnx`, `resnet8_int8.onnx`, `resnet8_uint8.onnx`)
