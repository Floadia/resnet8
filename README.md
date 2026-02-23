# ResNet8 CIFAR-10 Model Conversion and Quantization

Multi-framework model conversion from Keras to ONNX and PyTorch, with Post-Training Quantization (PTQ) support for model compression.

## Project Purpose

**Source:** MLCommons TinyMLPerf ResNet8 model trained on CIFAR-10

**Goal:** Enable cross-framework inference with high accuracy retention (>85%)

**Quantization:** Static PTQ with int8/uint8 support for reduced model size while maintaining accuracy

## Quick Results

| Model | Accuracy | Size | Reduction |
|-------|----------|------|-----------|
| FP32 baseline (ONNX) | 87.19% | 315KB | - |
| ONNX Runtime uint8 | 86.75% | 123KB | 61% |
| ONNX Runtime int8 | 85.58% | 123KB | 61% |
| PyTorch int8 | 85.68% | 165KB | 52% |

**Recommendation:** ONNX Runtime uint8 provides best accuracy-to-size ratio (-0.44% accuracy drop, 61% size reduction)

For detailed analysis, see [docs/QUANTIZATION_ANALYSIS.md](docs/QUANTIZATION_ANALYSIS.md)

## Project Structure

```
resnet8/
├── scripts/                    # Conversion and evaluation scripts
│   ├── convert.py              # Keras → ONNX conversion
│   ├── convert_pytorch.py      # ONNX → PyTorch conversion
│   ├── evaluate.py             # ONNX model evaluation
│   ├── evaluate_pytorch.py     # PyTorch model evaluation
│   ├── quantize_onnx.py        # ONNX Runtime PTQ
│   ├── quantize_pytorch.py     # PyTorch PTQ
│   └── calibration_utils.py    # Calibration data utilities
├── models/                     # Converted and quantized models
│   ├── resnet8.onnx           # FP32 ONNX baseline
│   ├── resnet8.pt             # FP32 PyTorch baseline
│   ├── resnet8_int8.onnx      # ONNX Runtime int8
│   ├── resnet8_uint8.onnx     # ONNX Runtime uint8
│   └── resnet8_int8.pt        # PyTorch int8
├── docs/                       # Analysis documents
│   └── QUANTIZATION_ANALYSIS.md
├── logs/                       # Conversion and evaluation logs
└── .planning/                  # Project planning artifacts
```

## Prerequisites

- **Python:** 3.12 or higher
- **Package manager:** uv (recommended) or pip
- **CIFAR-10 dataset:** Path to `cifar-10-batches-py` directory
- **Keras model:** (Optional) For conversion from scratch

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install tensorflow onnx onnxruntime torch torchvision onnx2torch numpy
pip install git+https://github.com/onnx/tensorflow-onnx.git
```

## Usage

### Convert Keras to ONNX

The `convert.py` script has hardcoded paths for the TinyMLPerf model. Edit the paths in the script or use it as-is if you have the reference project at `/mnt/ext1/references/tiny/`.

```bash
uv run python scripts/convert.py
```

Outputs: `models/resnet8.onnx`

### Evaluate ONNX Model

```bash
uv run python scripts/evaluate.py \
    models/resnet8.onnx \
    /path/to/cifar-10-batches-py
```

### Quantize with ONNX Runtime

Produces both int8 and uint8 quantized models:

```bash
uv run python scripts/quantize_onnx.py \
    --model models/resnet8.onnx \
    --data-dir /path/to/cifar-10-batches-py \
    --output-int8 models/resnet8_int8.onnx \
    --output-uint8 models/resnet8_uint8.onnx
```

Outputs: `models/resnet8_int8.onnx`, `models/resnet8_uint8.onnx`

### Convert to PyTorch

```bash
uv run python scripts/convert_pytorch.py \
    models/resnet8.onnx \
    models/resnet8.pt
```

Outputs: `models/resnet8.pt`

### Evaluate PyTorch Model

```bash
uv run python scripts/evaluate_pytorch.py \
    models/resnet8.pt \
    /path/to/cifar-10-batches-py
```

### Quantize with PyTorch

Produces int8 quantized model (uint8 not supported by fbgemm backend):

```bash
uv run python scripts/quantize_pytorch.py \
    --model models/resnet8.pt \
    --data-dir /path/to/cifar-10-batches-py \
    --output models/resnet8_int8.pt
```

Outputs: `models/resnet8_int8.pt`

## Architecture

**ResNet8 for CIFAR-10:**
- **Input:** 32×32×3 RGB images
- **Architecture:** 3 residual stacks (16→32→64 filters)
  - Initial: Conv2D(16, 3×3) → BatchNorm → ReLU
  - Stack 1: 2× residual blocks (16 filters, identity shortcut)
  - Stack 2: 2× residual blocks (32 filters, stride=2, 1×1 conv shortcut)
  - Stack 3: 2× residual blocks (64 filters, stride=2, 1×1 conv shortcut)
  - Head: Global Average Pooling → Dense(10, softmax)
- **Output:** 10-class probability distribution

**Training details:**
- Dataset: CIFAR-10 (60,000 training images, 10,000 test images)
- Regularization: L2 (1e-4) on conv kernels
- Initialization: He normal

## Quantization Details

### Calibration
- **Samples:** 1000 stratified samples from CIFAR-10 training set (100 per class)
- **Method:** MinMax calibration (ONNX Runtime), default observer (PyTorch)
- **Type:** Static post-training quantization (PTQ)

### Framework Support

**ONNX Runtime:**
- Supports both int8 and uint8 quantization
- QDQ format (QuantizeLinear/DequantizeLinear)
- Best accuracy: uint8 (86.75%)

**PyTorch:**
- int8 only (fbgemm backend limitation)
- FX graph mode quantization
- TorchScript serialization (JIT tracing)
- Accuracy: 85.68%

## Development

### Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI enforces these checks on all PRs.

```bash
# Check for lint errors
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-fix lint errors
uv run ruff check --fix .

# Auto-format code
uv run ruff format .
```

Run both checks before committing:

```bash
uv run ruff check . && uv run ruff format --check .
```

### Quantization Playground

Interactive Marimo notebook for exploring quantization parameters:

```bash
uv sync
marimo edit playground/quantization.py
```

### Agent Team (Codex Multi-Agent)

This repo includes a starter multi-agent setup inspired by harness engineering patterns:

- `explorer`: mapping and scoping (read-only)
- `implementer`: implementation (workspace-write)
- `reviewer`: regression/test-gap review (read-only)

See `docs/AGENT_TEAM.md` for usage prompts and workflow.

## License

This project uses the ResNet8 model from [MLCommons TinyMLPerf](https://github.com/mlcommons/tiny), which is licensed under Apache 2.0.

---

*Project version: 1.2.0*
*Last updated: 2026-01-28*
