# Phase 7: PyTorch Quantization - Research

**Researched:** 2026-01-28
**Domain:** Post-Training Quantization with PyTorch
**Confidence:** MEDIUM

## Summary

PyTorch quantization is in active transition from legacy eager mode APIs to modern torchao and PT2E (PyTorch 2 Export) approaches. As of 2026, PyTorch plans to delete torch.ao.quantization in version 2.10, with PT2E quantization using X86InductorQuantizer as the recommended long-term supported path for static quantization on CPU. However, eager mode quantization remains available (beta status) and is simpler for models with standard layer types.

For ResNet8 quantization, two viable approaches exist: (1) Eager mode static quantization using torch.ao.quantization (simpler, well-documented, works with standard PyTorch models), or (2) PT2E quantization with X86InductorQuantizer (future-proof, requires PyTorch ≥2.6, better performance on Intel CPUs). Given the onnx2torch-converted model's uncertain compatibility with PT2E export (export() requires symbolically traceable models), eager mode is the safer starting point with PT2E as a potential optimization if needed.

The critical challenge for this phase is that the existing resnet8.pt model was created via onnx2torch conversion, which produces nn.Module layers that may not be fully compatible with PyTorch quantization requirements (module fusion patterns, FloatFunctional for skip connections, QuantStub/DeQuantStub wrappers). The model will likely need structural modifications before quantization can succeed.

**Primary recommendation:** Start with eager mode static quantization (torch.ao.quantization) for ResNet8 using the existing onnx2torch-converted model. Inspect model structure to identify required modifications (add QuantStub/DeQuantStub, replace skip connection additions with FloatFunctional.add, configure fusion patterns). Use fbgemm backend with default qconfig (quint8 activations, qint8 weights), calibrate with existing 1000-sample calibration data, and compare against ONNX Runtime results (uint8: 86.75%, int8: 85.58%) to validate approach.

## Standard Stack

The established libraries/tools for PyTorch quantization:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.6+ | Built-in quantization APIs | Native quantization support in torch.ao.quantization (eager mode) and torch.export (PT2E) |
| torchao | 0.15+ | Modern quantization toolkit | Long-term supported path replacing torch.ao.quantization, PT2E quantization flow |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | Any | Calibration data handling | Loading and preprocessing CIFAR-10 calibration samples |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Eager mode (torch.ao.quantization) | PT2E with X86InductorQuantizer | PT2E is future-proof and faster but requires PyTorch ≥2.6, export() compatibility, and more complex setup |
| Static quantization | Dynamic quantization | Dynamic is faster to implement (no calibration) but only quantizes weights, lower accuracy, slower inference |
| fbgemm backend | x86 backend (PyTorch 2.0+) | x86 backend offers better performance leveraging oneDNN but fbgemm is simpler and proven |

**Installation:**
```bash
# PyTorch 2.6+ already installed
# Optional: torchao for PT2E quantization (if needed)
pip install torchao
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── quantize_pytorch.py        # PyTorch static quantization script
├── calibration_utils.py       # Existing: provides load_calibration_data()
└── evaluate_pytorch.py        # Existing: evaluates PyTorch models (works with quantized)

models/
├── resnet8.pt                 # Source FP32 model (onnx2torch-converted)
├── resnet8_quantized_int8.pt  # Quantized int8 model (output)
└── resnet8_quantized_uint8.pt # Quantized uint8 model if supported (output)
```

### Pattern 1: Eager Mode Static Quantization Workflow
**What:** Four-step process: fusion → prepare → calibrate → convert
**When to use:** Standard PyTorch models with known layer types, simpler than PT2E
**Example:**
```python
# Source: PyTorch official tutorial + Lei Mao blog
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules

# Step 1: Model preparation
model = load_model('models/resnet8.pt')
model.eval()  # CRITICAL: must be in eval mode for BatchNorm fusion
model.to('cpu')  # Static quantization only supports CPU

# Step 2: Layer fusion (Conv+BN+ReLU patterns)
# IMPORTANT: Must specify exact module paths in model
model = fuse_modules(model, [
    ['conv1', 'bn1', 'relu1'],  # Input layer
    ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],  # Residual blocks
    ['layer1.0.conv2', 'layer1.0.bn2'],  # No ReLU after second conv
    # ... additional fusion patterns based on ResNet8 architecture
])

# Step 3: Configure quantization
model.qconfig = get_default_qconfig('fbgemm')  # quint8 activations, qint8 weights
prepared_model = prepare(model, inplace=False)

# Step 4: Calibrate with representative data
for data, _ in calibration_loader:
    prepared_model(data)

# Step 5: Convert to quantized model
quantized_model = convert(prepared_model, inplace=False)

# Save quantized model
torch.save({'model': quantized_model}, 'models/resnet8_quantized_int8.pt')
```

### Pattern 2: PT2E Static Quantization Workflow (Alternative)
**What:** Modern export-based quantization with X86InductorQuantizer
**When to use:** When model is compatible with torch.export, need best CPU performance, PyTorch ≥2.6
**Example:**
```python
# Source: torchao official tutorial
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq

# Step 1: Export model to FX graph (may fail with onnx2torch models)
model.eval()
example_input = torch.randn(1, 32, 32, 3)  # NHWC format
exported_model = export(model, (example_input,)).module()

# Step 2: Configure quantizer
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

# Step 3: Prepare and calibrate
prepared_model = prepare_pt2e(exported_model, quantizer)
for data, _ in calibration_loader:
    prepared_model(data)

# Step 4: Convert to quantized model
quantized_model = convert_pt2e(prepared_model)

# Step 5: Compile for inductor backend
optimized_model = torch.compile(quantized_model)
```

### Pattern 3: Model Modifications for Quantization
**What:** Add QuantStub/DeQuantStub wrappers and FloatFunctional for skip connections
**When to use:** Required for eager mode quantization, especially with ResNet architectures
**Example:**
```python
# Source: PyTorch quantization tutorials + Lei Mao blog
from torch.nn.quantized import FloatFunctional

class QuantizableResNet8(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Add quantization stubs for input/output
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Copy layers from original model
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        # ... other layers

        # Replace skip connection additions with FloatFunctional
        self.skip_add = FloatFunctional()

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # Forward pass
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        # ... more layers

        # CRITICAL: Use FloatFunctional for skip connections
        x = self.skip_add.add(x, identity)

        # Dequantize output
        x = self.dequant(x)
        return x
```

### Pattern 4: Calibration Data Loading
**What:** Load calibration samples in batches using existing calibration_utils.py
**When to use:** Both eager mode and PT2E quantization require calibration
**Example:**
```python
# Source: Existing calibration_utils.py + PyTorch tutorials
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from calibration_utils import load_calibration_data

# Load 1000 stratified samples (100 per class)
np.random.seed(42)
images, labels, _ = load_calibration_data(
    data_dir='/path/to/cifar-10-batches-py',
    samples_per_class=100
)

# CRITICAL: Convert NHWC to NCHW if needed for PyTorch
# Check model input format first - onnx2torch may preserve NHWC
if model_expects_nchw:
    images = images.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

# Create DataLoader
dataset = TensorDataset(
    torch.from_numpy(images),
    torch.from_numpy(labels)
)
calibration_loader = DataLoader(dataset, batch_size=32, shuffle=False)
```

### Anti-Patterns to Avoid
- **Using + operator for skip connections:** Must use FloatFunctional.add() for proper quantization statistics collection, otherwise input/output quantization will fail
- **Reusing ReLU modules:** Each ReLU must have unique name (relu1, relu2) for fusion, cannot define `relu = nn.ReLU()` and reuse it
- **Forgetting QuantStub/DeQuantStub:** Input and output must be wrapped with QuantStub and DeQuantStub for proper quantization boundaries
- **Fusion before eval mode:** Must call model.eval() before fusing modules, otherwise BatchNorm fusion will fail
- **Wrong calibration data format:** Must match model's expected input format (NCHW vs NHWC), preprocessing mismatches cause catastrophic accuracy loss

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Quantization parameter calculation | Manual scale/zero-point computation from calibration stats | get_default_qconfig() with MinMaxObserver/HistogramObserver | Observers handle edge cases (zero ranges, outliers), per-channel complexity, proper asymmetric quantization |
| Module fusion detection | Manual graph traversal to find Conv+BN+ReLU patterns | fuse_modules() with explicit fusion lists | Handles weight/bias folding correctness, BatchNorm statistics integration, activation function fusion |
| Skip connection quantization | Custom quantized addition ops | FloatFunctional.add() | Properly inserts quantization observers, handles scale/zero-point matching between branches |
| Model structure wrapping | Manual QuantizeLinear/DequantizeLinear insertion | QuantStub and DeQuantStub wrappers | Integrates with prepare/convert workflow, handles qconfig propagation automatically |

**Key insight:** PyTorch quantization requires specific module types and patterns that were designed with native PyTorch models in mind. Converted models (onnx2torch) may not follow these patterns, requiring structural modifications rather than just parameter tuning.

## Common Pitfalls

### Pitfall 1: onnx2torch Model Incompatibility with Quantization
**What goes wrong:** onnx2torch-converted models use generic nn.Module layers that may not match PyTorch's quantization expectations (layer names, fusion patterns, functional ops), causing fusion failures, missing observers, or export errors
**Why it happens:** onnx2torch recreates ONNX operations as PyTorch modules but doesn't guarantee compatibility with PyTorch's quantization framework which expects specific patterns from torchvision or native training
**How to avoid:** Inspect converted model structure first (print(model)), verify layer types match expected patterns (nn.Conv2d, nn.BatchNorm2d, nn.ReLU as separate modules), be prepared to restructure model by wrapping in QuantizableResNet8 class with proper module names and skip connection handling
**Warning signs:** fuse_modules() raises KeyError for module names, prepare() doesn't insert observers, export() fails with "not symbolically traceable" error, model has unexpected layer types (custom ONNX ops)

### Pitfall 2: NHWC vs NCHW Data Format Mismatch
**What goes wrong:** onnx2torch may preserve original ONNX model's NHWC (batch, height, width, channels) format while PyTorch quantization typically expects NCHW (batch, channels, height, width), causing shape mismatches or silent accuracy degradation
**Why it happens:** ONNX models from TensorFlow/Keras use NHWC layout, onnx2torch conversion may preserve this for compatibility, but PyTorch Conv2d and quantization observers expect NCHW by default
**How to avoid:** Check model's expected input shape by running dummy inference with both formats, verify evaluate_pytorch.py input format (currently uses NHWC: 32, 32, 3), transpose calibration data if needed (images.transpose(0, 3, 1, 2) for NHWC→NCHW)
**Warning signs:** Model accepts input but produces garbage predictions after quantization, shape mismatch errors during calibration, accuracy drops to near-random levels (<20%)

### Pitfall 3: fbgemm Backend uint8 Activation Support Uncertainty
**What goes wrong:** PyTorch fbgemm backend documentation and sources conflict on whether uint8 activations (quint8) are fully supported vs only int8 (qint8) for weights and activations
**Why it happens:** PyTorch quantization evolved through multiple APIs (eager mode, FX, PT2E), backend capabilities changed over versions, and documentation may refer to different quantization schemes (per-tensor vs per-channel, static vs dynamic)
**How to avoid:** Start with default qconfig (get_default_qconfig('fbgemm')) which uses quint8 activations and qint8 weights, test both configurations, verify accuracy, document which works better, accept that uint8-only quantization may not be possible with fbgemm (unlike ONNX Runtime which clearly supports U8U8)
**Warning signs:** Quantization succeeds but accuracy is significantly worse than ONNX Runtime uint8 (86.75%), model runs but slower than FP32, backend raises "unsupported dtype" errors

### Pitfall 4: Module Fusion Pattern Specification Errors
**What goes wrong:** fuse_modules() requires exact module path lists like [['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu']], but onnx2torch models may have different naming schemes or nested structures, causing fusion to silently skip or raise KeyError
**Why it happens:** Fusion patterns are specified as string paths relative to model root, onnx2torch may generate different module hierarchies than expected, and fusion silently continues if pattern doesn't match
**How to avoid:** Print full model structure (for name, module in model.named_modules(): print(name)), verify each fusion pattern exists, start with no fusion to establish baseline, add fusion patterns incrementally and verify each works
**Warning signs:** Quantized model has same size as FP32 (fusion failed), accuracy severely degraded (>10% drop), fuse_modules() completes but no performance improvement

### Pitfall 5: Insufficient Calibration with Small Batches
**What goes wrong:** Using too few calibration samples (<100 total) or wrong batch structure (single large batch vs many small batches) causes observers to collect poor statistics, resulting in suboptimal scale/zero-point and 5-10% accuracy loss
**Why it happens:** Observers need to see diverse activation ranges across multiple batches to compute robust quantization parameters, single-batch calibration misses temporal variation
**How to avoid:** Use 1000 samples (100 per class) from existing calibration_utils.py, batch_size=32 in DataLoader (typical for calibration), run full calibration loop (30-32 batches), verify observer statistics are reasonable (check min/max ranges)
**Warning signs:** Quantized accuracy drops >5% from baseline, specific classes show severe degradation, quantization parameters have suspiciously narrow ranges (scale ≈ 1.0, zero_point = 0)

### Pitfall 6: Forgetting to Set Model to Eval Mode
**What goes wrong:** Calling prepare() or fuse_modules() on a model in training mode causes BatchNorm layers to maintain running statistics mode instead of freezing, leading to non-deterministic quantization and wrong fusion behavior
**Why it happens:** BatchNorm behaves differently in train vs eval mode (running stats vs batch stats), fusion requires eval mode to fold BN parameters into Conv weights
**How to avoid:** Always call model.eval() before any quantization operations, verify with model.training == False, keep model in eval throughout fusion→prepare→calibrate→convert pipeline
**Warning signs:** Quantized model produces different outputs on same input across runs, BatchNorm layers not fused (model still has separate BN modules), calibration accuracy varies significantly

## Code Examples

Verified patterns from official sources:

### Complete Eager Mode Quantization Script Structure
```python
#!/usr/bin/env python3
"""Quantize ResNet8 PyTorch model using static quantization (eager mode)."""

import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules

# Import existing utilities
import sys
sys.path.append('scripts')
from calibration_utils import load_calibration_data


def load_pytorch_model(model_path: str) -> torch.nn.Module:
    """Load PyTorch model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    model.eval()  # CRITICAL: must be in eval mode
    return model


def inspect_model_structure(model: torch.nn.Module):
    """Print model structure to identify fusion patterns."""
    print("Model Structure:")
    print("=" * 60)
    for name, module in model.named_modules():
        print(f"{name:40s} {type(module).__name__}")
    print("=" * 60)


def create_calibration_loader(data_dir: str, samples_per_class: int = 100,
                               batch_size: int = 32) -> DataLoader:
    """Create calibration DataLoader from existing utilities."""
    # Load stratified calibration data
    np.random.seed(42)
    images, labels, _ = load_calibration_data(data_dir, samples_per_class)

    print(f"Loaded {len(images)} calibration samples")
    print(f"  Shape: {images.shape}")
    print(f"  dtype: {images.dtype}")
    print(f"  Range: [{images.min():.1f}, {images.max():.1f}]")

    # Convert to tensors
    dataset = TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Created DataLoader with batch_size={batch_size}, {len(loader)} batches")

    return loader


def quantize_model_eager(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    fusion_patterns: list = None,
) -> torch.nn.Module:
    """Quantize model using eager mode static quantization.

    Args:
        model: FP32 PyTorch model in eval mode
        calibration_loader: DataLoader with calibration samples
        fusion_patterns: List of module name lists for fusion (e.g., [['conv1', 'bn1', 'relu1']])

    Returns:
        Quantized PyTorch model
    """
    # Step 1: Layer fusion (if patterns provided)
    if fusion_patterns:
        print(f"\nFusing {len(fusion_patterns)} layer patterns...")
        model = fuse_modules(model, fusion_patterns, inplace=True)
        print("Fusion complete")
    else:
        print("\nSkipping fusion (no patterns provided)")

    # Step 2: Configure quantization
    print("\nConfiguring quantization (fbgemm backend)...")
    model.qconfig = get_default_qconfig('fbgemm')
    print(f"  Activation observer: {model.qconfig.activation}")
    print(f"  Weight observer: {model.qconfig.weight}")

    # Step 3: Prepare model (insert observers)
    print("\nPreparing model (inserting observers)...")
    prepared_model = prepare(model, inplace=True)
    print("Model prepared")

    # Step 4: Calibration
    print(f"\nCalibrating with {len(calibration_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            prepared_model(data)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(calibration_loader)} batches")
    print("Calibration complete")

    # Step 5: Convert to quantized model
    print("\nConverting to quantized model...")
    quantized_model = convert(prepared_model, inplace=True)
    print("Conversion complete")

    return quantized_model


def main():
    parser = argparse.ArgumentParser(description="Quantize ResNet8 PyTorch model")
    parser.add_argument(
        "--model",
        default="models/resnet8.pt",
        help="Path to FP32 PyTorch model"
    )
    parser.add_argument(
        "--output",
        default="models/resnet8_quantized_int8.pt",
        help="Path for quantized model output"
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py",
        help="CIFAR-10 data directory"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Calibration samples per class (default: 100, total: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Calibration batch size (default: 32)"
    )
    parser.add_argument(
        "--inspect-only",
        action='store_true',
        help="Only inspect model structure without quantizing"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PYTORCH STATIC QUANTIZATION (EAGER MODE)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_pytorch_model(args.model)
    print("Model loaded")

    # Inspect structure
    inspect_model_structure(model)

    if args.inspect_only:
        print("\nInspection complete (--inspect-only flag set)")
        return

    # Create calibration loader
    print(f"\nLoading calibration data from: {args.data_dir}")
    calibration_loader = create_calibration_loader(
        args.data_dir,
        args.samples_per_class,
        args.batch_size
    )

    # TODO: Define fusion patterns based on model structure
    # CRITICAL: Must match actual module names from inspect_model_structure()
    # Example for standard ResNet:
    # fusion_patterns = [
    #     ['conv1', 'bn1', 'relu'],
    #     ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu1'],
    #     ['layer1.0.conv2', 'layer1.0.bn2'],
    #     # ... etc
    # ]
    fusion_patterns = None  # Start without fusion to establish baseline

    # Quantize model
    quantized_model = quantize_model_eager(
        model,
        calibration_loader,
        fusion_patterns
    )

    # Save quantized model
    print(f"\nSaving quantized model to: {args.output}")
    torch.save({'model': quantized_model}, args.output)
    print("Quantized model saved")

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"Output: {args.output}")
    print("\nNext step: Evaluate quantized model with scripts/evaluate_pytorch.py")


if __name__ == "__main__":
    main()
```

### Evaluation Command (Existing Script Works)
```bash
# Existing evaluate_pytorch.py works transparently with quantized models
python scripts/evaluate_pytorch.py --model models/resnet8_quantized_int8.pt

# Compare with FP32 baseline (87.19% expected from onnx2torch model)
python scripts/evaluate_pytorch.py --model models/resnet8.pt

# Compare with ONNX Runtime quantized results
# ONNX Runtime uint8: 86.75%
# ONNX Runtime int8: 85.58%
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Eager mode quantization (torch.ao.quantization) | PT2E quantization (torchao) | Migration plan announced 2025 | PT2E is long-term supported path, eager mode deprecated in PyTorch 2.10 but still usable in 2.6-2.9 |
| FX graph mode quantization | PT2E quantization | Same migration (2025) | FX graph mode in maintenance, users should migrate to PT2E |
| fbgemm backend | x86 backend (PyTorch 2.0+) | PyTorch 2.0 (2023) | x86 backend offers better performance using oneDNN but fbgemm still widely used and supported |
| Manual observer configuration | get_default_qconfig() | Long-standing best practice | Default configs (MinMaxObserver for weights, HistogramObserver for activations) work well for CNNs |

**Deprecated/outdated:**
- **torch.ao.quantization APIs:** Scheduled for deletion in PyTorch 2.10 (late 2025/early 2026), migrate to torchao or PT2E
- **FX graph mode quantization:** In maintenance mode, no new features, use PT2E instead
- **Manual QuantStub placement everywhere:** PT2E export handles quantization boundaries automatically

## Open Questions

Things that couldn't be fully resolved:

1. **Does onnx2torch model structure support PyTorch quantization?**
   - What we know: onnx2torch creates nn.Module layers but may not follow PyTorch conventions (module naming, fusion patterns, skip connection handling)
   - What's unclear: Whether resnet8.pt has standard layer types (nn.Conv2d, nn.BatchNorm2d, nn.ReLU as separate modules), correct module hierarchy for fusion, or needs complete restructuring
   - Recommendation: Run --inspect-only first to print model structure, compare against standard torchvision ResNet patterns, be prepared to wrap model in QuantizableResNet8 class if structure is incompatible

2. **What is ResNet8's input format (NCHW vs NHWC)?**
   - What we know: evaluate_pytorch.py currently uses NHWC (32, 32, 3), calibration_utils.py outputs NHWC, ONNX model likely preserves original TensorFlow/Keras NHWC layout
   - What's unclear: Whether onnx2torch conversion automatically transposes to NCHW (PyTorch standard), or preserves NHWC and expects that format
   - Recommendation: Test both formats in quantization script, verify model accepts NHWC without transpose, transpose calibration data to NCHW if model expects it (check with dummy inference)

3. **Does fbgemm backend support uint8-only quantization?**
   - What we know: fbgemm default qconfig uses quint8 activations + qint8 weights, PyTorch documentation mentions qint8/quint8/qint32 dtypes, ONNX Runtime clearly supports U8U8 (uint8 activations and weights)
   - What's unclear: Whether fbgemm can do uint8 weights (to match ONNX Runtime's uint8 quantization which achieved 86.75% vs int8's 85.58%), or if uint8 is only supported for activations
   - Recommendation: Start with default qconfig (quint8 activations, qint8 weights), test if accuracy is similar to ONNX Runtime uint8, document that "uint8 model" in PyTorch means quint8 activations (not U8U8 like ONNX Runtime)

4. **Should we use eager mode or PT2E quantization?**
   - What we know: Eager mode is simpler, well-documented, but deprecated in PyTorch 2.10; PT2E is future-proof, faster on Intel CPUs, but requires PyTorch ≥2.6 and export() compatibility
   - What's unclear: Whether onnx2torch model can be exported via torch.export (requires symbolically traceable forward pass), or if we need to use eager mode due to compatibility issues
   - Recommendation: Start with eager mode (simpler, more likely to work with converted model), validate against ONNX Runtime results, consider PT2E as optimization if eager mode succeeds and we need better performance

5. **What fusion patterns does ResNet8 use?**
   - What we know: Standard ResNet has Conv+BN+ReLU in input layer and residual blocks, but ResNet8 is a custom smaller architecture (not in torchvision)
   - What's unclear: Exact layer structure, module names, which layers have ReLU activations, whether skip connections exist (residual blocks)
   - Recommendation: Use --inspect-only mode to print full model structure, identify fusion patterns manually, start without fusion to establish baseline accuracy, add fusion incrementally

## Sources

### Primary (HIGH confidence)
- [PyTorch Quantization Documentation (2.10)](https://docs.pytorch.org/docs/stable/quantization.html) - Official quantization overview, API status
- [torchao Static Quantization Tutorial](https://docs.pytorch.org/ao/stable/static_quantization.html) - Updated Jan 2026
- [PT2E Quantization Tutorial](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html) - Updated Jan 2026
- [PyTorch Migration Plan Discussion](https://dev-discuss.pytorch.org/t/torch-ao-quantization-migration-plan/2810) - Official deprecation timeline
- [PyTorch Quantization Flow Support](https://dev-discuss.pytorch.org/t/clarification-of-pytorch-quantization-flow-support-in-pytorch-and-torchao/2809) - Status clarification 2025

### Secondary (MEDIUM confidence)
- [Lei Mao's PyTorch Static Quantization Guide](https://leimao.github.io/blog/PyTorch-Static-Quantization/) - Detailed ResNet18 example with fusion patterns, verified approach
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/) - Official blog with best practices
- [X86 Inductor Quantization Tutorial](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_x86_inductor.html) - PT2E with X86InductorQuantizer
- [PyTorch Static Quantization Tutorial (Beta)](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html) - Official eager mode tutorial
- [onnx2torch GitHub Repository](https://github.com/ENOT-AutoDL/onnx2torch) - v1.5.15 released Aug 2024, conversion limitations

### Tertiary (LOW confidence - needs validation)
- [INT8 Quantization for x86 CPU Blog](https://pytorch.org/blog/int8-quantization/) - fbgemm backend capabilities (older article, may be outdated)
- [Neural Network Quantization Tutorial](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/) - Community tutorial, ResNet CIFAR-10 example (not officially verified)
- [onnx2torch Quantization Compatibility](https://discuss.pytorch.org/t/converting-quantized-models-from-pytorch-to-onnx/84855) - Forum discussions about ONNX↔PyTorch quantization (anecdotal)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch built-in quantization well-documented, torchao officially supported
- Architecture patterns: MEDIUM - Eager mode patterns verified in official tutorials, but onnx2torch model compatibility uncertain
- Pitfalls: MEDIUM - Common issues documented in official sources (FloatFunctional, fusion, eval mode), but onnx2torch-specific issues based on inference
- Expected accuracy: LOW - No direct ResNet8 benchmarks found, extrapolating from ResNet18/50 results (0.5-2% loss typical), but onnx2torch conversion adds uncertainty

**Research date:** 2026-01-28
**Valid until:** 30 days (PyTorch quantization APIs changing rapidly with 2.10 release approaching, eager mode deprecation imminent)
