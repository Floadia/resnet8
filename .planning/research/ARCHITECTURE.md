# Architecture Research: PTQ Integration

**Project:** ResNet8 CIFAR-10 PTQ Evaluation
**Domain:** Post-Training Quantization (Static)
**Researched:** 2026-01-28
**Confidence:** HIGH

## Executive Summary

PTQ integration follows a **quantize-then-evaluate pattern** that extends existing evaluation scripts with calibration and quantization steps. Both ONNX Runtime and PyTorch use similar workflows: (1) prepare calibration data, (2) quantize model with calibration, (3) evaluate quantized model using existing evaluation infrastructure. The architecture maintains clean separation between quantization (one-time conversion) and evaluation (repeated validation), mirroring the existing conversion/evaluation pattern from v1.0-v1.1.

## Integration Points with Existing Components

### ONNX Runtime Quantization Integration

| Existing Component | Integration Point | How PTQ Integrates |
|-------------------|-------------------|-------------------|
| `scripts/evaluate.py` | CIFAR-10 loading (`load_cifar10_test()`) | Reuse for calibration data preparation |
| `scripts/evaluate.py` | Evaluation logic (`evaluate_model()`) | Identical interface - quantized .onnx works with same code |
| `models/resnet8.onnx` | Input model | Source for quantization - produces `resnet8_int8.onnx`, `resnet8_uint8.onnx` |
| CIFAR-10 test set | Test data | Subset (100-200 images) becomes calibration data |

**Key insight:** ONNX Runtime quantization produces standard .onnx files that work with existing `onnxruntime.InferenceSession` - no changes to evaluation script needed.

### PyTorch Quantization Integration

| Existing Component | Integration Point | How PTQ Integrates |
|-------------------|-------------------|-------------------|
| `scripts/evaluate_pytorch.py` | CIFAR-10 loading (`load_cifar10_test()`) | Reuse for calibration data preparation |
| `scripts/evaluate_pytorch.py` | Model loading (`load_pytorch_model()`) | Minor change - load quantized .pt differently |
| `scripts/evaluate_pytorch.py` | Evaluation logic (`evaluate_model()`) | Identical interface - quantized model is still `torch.nn.Module` |
| `models/resnet8.pt` | Input model | Source for quantization - produces `resnet8_int8.pt`, `resnet8_uint8.pt` |
| CIFAR-10 test set | Test data | Subset (100-200 images) becomes calibration data |

**Key insight:** PyTorch quantized models are still `torch.nn.Module` instances, so existing evaluation code works with minor model loading changes.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PTQ Evaluation Architecture                      │
└─────────────────────────────────────────────────────────────────────┘

EXISTING (v1.0-v1.1):
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ FP32 Models  │      │ Evaluation   │      │  CIFAR-10    │
│              │─────▶│  Scripts     │◄─────│  Test Data   │
│ .onnx, .pt   │      │              │      │  (10K imgs)  │
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                             ▼
                      Accuracy Reports


NEW (v1.2 PTQ):
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ FP32 Models  │      │ Calibration  │      │  CIFAR-10    │
│              │      │     Data     │◄─────│  Subset      │
│ .onnx, .pt   │      │  (100-200)   │      │ (from 10K)   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │
       │              ┌──────┘
       │              │
       ▼              ▼
┌──────────────────────────────┐
│  Quantization Scripts        │
│  - quantize_onnx.py          │
│  - quantize_pytorch.py       │
└──────────────────────────────┘
       │
       │ Produces quantized models
       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ INT8 Models  │      │ Evaluation   │      │  CIFAR-10    │
│              │─────▶│  Scripts     │◄─────│  Full Test   │
│ *_int8.onnx  │      │  (REUSED)    │      │  (10K imgs)  │
│ *_int8.pt    │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                             ▼
                   Quantized Accuracy Reports
```

## New Components Needed

### 1. Calibration Data Utility (`scripts/calibration_utils.py`)

**Purpose:** Prepare representative calibration dataset from CIFAR-10.

**Functionality:**
- Load CIFAR-10 test batch (reuse existing loading logic)
- Sample 100-200 images (stratified sampling across classes)
- Return calibration subset as numpy arrays
- Configurable sample size and random seed for reproducibility

**Why separate:** Both ONNX Runtime and PyTorch quantization scripts need calibration data - avoid duplication.

**Interface:**
```python
def get_calibration_data(
    data_dir: str,
    num_samples: int = 200,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) subset for calibration."""
    pass
```

### 2. ONNX Runtime Quantization Script (`scripts/quantize_onnx.py`)

**Purpose:** Apply static quantization to ONNX model.

**Workflow:**
1. Load FP32 ONNX model (`models/resnet8.onnx`)
2. Create `CalibrationDataReader` using calibration data
3. Call `quantize_static()` with int8 or uint8 configuration
4. Save quantized model (`models/resnet8_int8.onnx`)
5. Repeat for uint8 variant

**Key operations:**
- Pre-processing with `shape_inference.quant_pre_process()` (optional but recommended)
- Quantization with `quantize_static(model_path, output_path, calibration_data_reader, ...)`
- Configuration: `QuantType.QInt8` or `QuantType.QUInt8` for activations/weights

**Dependencies:**
- `onnxruntime.quantization` module
- Calibration utils for data preparation
- Original ONNX model

### 3. PyTorch Quantization Script (`scripts/quantize_pytorch.py`)

**Purpose:** Apply static quantization to PyTorch model.

**Workflow:**
1. Load FP32 PyTorch model (`models/resnet8.pt`)
2. Prepare calibration data loader
3. Set quantization configuration (qconfig)
4. Prepare model for quantization (`prepare()` or `prepare_pt2e()`)
5. Run calibration (forward pass with calibration data)
6. Convert to quantized model (`convert()` or `convert_pt2e()`)
7. Save quantized model (`models/resnet8_int8.pt`)
8. Repeat for uint8 variant

**Key operations:**
- Model export: `torch.export.export()` (PT2E path, recommended)
- Preparation: `prepare_pt2e()` inserts observers
- Calibration: Forward pass with representative data
- Conversion: `convert_pt2e()` produces quantized model

**Dependencies:**
- `torch.ao.quantization` or `torchao` module
- Calibration utils for data preparation
- Original PyTorch model

### 4. Quantized Model Evaluation (Reuse Existing)

**No new component needed** - existing evaluation scripts work with quantized models.

**ONNX Runtime:**
- `scripts/evaluate.py --model models/resnet8_int8.onnx` works unchanged
- `onnxruntime.InferenceSession` loads quantized .onnx transparently

**PyTorch:**
- `scripts/evaluate_pytorch.py --model models/resnet8_int8.pt` works with minor changes
- Quantized model is still `torch.nn.Module`, just with quantized operations

**Modification needed (PyTorch only):**
- Update `load_pytorch_model()` to handle quantized models if serialization format differs
- Likely minimal - quantized models can be saved as standard checkpoints

## Data Flow

### ONNX Runtime Quantization Flow

```
1. Load FP32 Model
   ├─ Input: models/resnet8.onnx
   └─ Action: Pre-process with quant_pre_process() (optional)

2. Prepare Calibration Data
   ├─ Input: CIFAR-10 test batch (10,000 images)
   ├─ Action: Sample 200 images (20 per class, stratified)
   └─ Output: (200, 32, 32, 3) float32 array, range [0, 255]

3. Create CalibrationDataReader
   ├─ Input: Calibration images
   ├─ Action: Wrap in CalibrationDataReader class
   └─ Output: Iterator yielding {input_name: batch_data}

4. Quantize Model (INT8)
   ├─ Input: Pre-processed .onnx, CalibrationDataReader
   ├─ Action: quantize_static(..., weight_type=QInt8, activation_type=QInt8)
   ├─ Calibration: Run inference to collect activation statistics
   └─ Output: models/resnet8_int8.onnx

5. Quantize Model (UINT8)
   ├─ Input: Pre-processed .onnx, CalibrationDataReader
   ├─ Action: quantize_static(..., weight_type=QUInt8, activation_type=QUInt8)
   └─ Output: models/resnet8_uint8.onnx

6. Evaluate Quantized Models
   ├─ Input: resnet8_int8.onnx, resnet8_uint8.onnx, full test set (10K)
   ├─ Action: scripts/evaluate.py --model models/resnet8_int8.onnx
   └─ Output: Accuracy reports (compare to 87.19% baseline)
```

### PyTorch Quantization Flow

```
1. Load FP32 Model
   ├─ Input: models/resnet8.pt
   ├─ Action: torch.load(), set to eval mode
   └─ Output: Model ready for quantization

2. Prepare Calibration Data
   ├─ Input: CIFAR-10 test batch (10,000 images)
   ├─ Action: Sample 200 images, convert to torch.Tensor
   └─ Output: DataLoader with (200, 32, 32, 3) tensors

3. Export Model (PT2E)
   ├─ Input: FP32 model
   ├─ Action: torch.export.export(model, example_inputs)
   └─ Output: Exported GraphModule

4. Prepare for Quantization
   ├─ Input: Exported model, quantizer configuration
   ├─ Action: prepare_pt2e(model, quantizer)
   ├─ Effect: Inserts observers after activations
   └─ Output: Model with observers

5. Calibrate (INT8)
   ├─ Input: Prepared model, calibration DataLoader
   ├─ Action: Forward pass through calibration data
   ├─ Effect: Observers collect activation statistics
   └─ Output: Calibrated model

6. Convert to Quantized (INT8)
   ├─ Input: Calibrated model
   ├─ Action: convert_pt2e(model)
   └─ Output: models/resnet8_int8.pt

7. Repeat for UINT8
   ├─ Configure: Use quint8 dtypes if supported
   └─ Output: models/resnet8_uint8.pt

8. Evaluate Quantized Models
   ├─ Input: resnet8_int8.pt, resnet8_uint8.pt, full test set (10K)
   ├─ Action: scripts/evaluate_pytorch.py --model models/resnet8_int8.pt
   └─ Output: Accuracy reports (compare to 87.19% baseline)
```

## Suggested Build Order

### Phase 1: Calibration Infrastructure (Foundation)

**Why first:** Both ONNX Runtime and PyTorch quantization need calibration data. Building this first enables parallel development of quantization scripts.

**Deliverables:**
- `scripts/calibration_utils.py` with `get_calibration_data()` function
- Unit test: Verify stratified sampling produces balanced class distribution
- Verify: 200 samples, shape (200, 32, 32, 3), raw pixel values [0, 255]

**Dependencies:** CIFAR-10 test batch (already available)

**Effort:** Low (reuse existing loading logic from evaluate.py)

---

### Phase 2: ONNX Runtime Quantization (Simpler Path)

**Why second:** ONNX Runtime quantization is simpler - no model export step, quantized models use same evaluation script unchanged.

**Deliverables:**
- `scripts/quantize_onnx.py` script
- Quantized models: `models/resnet8_int8.onnx`, `models/resnet8_uint8.onnx`
- Evaluation results for both quantized models

**Key tasks:**
1. Implement `CalibrationDataReader` class
2. Add pre-processing step with `quant_pre_process()`
3. Call `quantize_static()` with INT8 configuration
4. Verify quantized model loads with `onnxruntime.InferenceSession`
5. Run evaluation with existing `scripts/evaluate.py`
6. Repeat for UINT8

**Dependencies:**
- Phase 1 (calibration data)
- `onnxruntime.quantization` module
- Original `models/resnet8.onnx`

**Effort:** Medium (CalibrationDataReader implementation, quantization configuration)

---

### Phase 3: PyTorch Quantization (Complex Path)

**Why third:** PyTorch quantization requires model export (PT2E), more complex API, potential model loading changes.

**Deliverables:**
- `scripts/quantize_pytorch.py` script
- Quantized models: `models/resnet8_int8.pt`, `models/resnet8_uint8.pt`
- Updated `scripts/evaluate_pytorch.py` if needed for loading quantized models
- Evaluation results for both quantized models

**Key tasks:**
1. Export model with `torch.export.export()`
2. Configure quantizer (decide on backend: x86, qnnpack, etc.)
3. Prepare model with `prepare_pt2e()`
4. Run calibration (forward pass with calibration data)
5. Convert to quantized with `convert_pt2e()`
6. Test serialization/deserialization of quantized model
7. Update evaluation script if model loading differs
8. Run evaluation
9. Repeat for UINT8 (if supported)

**Dependencies:**
- Phase 1 (calibration data)
- `torch.ao.quantization` or `torchao` module
- Original `models/resnet8.pt`

**Effort:** High (PT2E export, quantizer configuration, potential evaluation script changes)

**Risk factors:**
- PyTorch quantization API changed significantly in 2.x (PT2E vs eager mode)
- UINT8 support may be limited (PyTorch traditionally uses INT8 on CPU)
- Quantized model serialization format may differ from FP32

---

### Phase 4: Comparison and Documentation

**Why last:** Requires all quantized models evaluated to produce comparison matrix.

**Deliverables:**
- Accuracy comparison table: FP32 vs INT8 vs UINT8 for both frameworks
- Model size comparison
- Documentation of accuracy deltas
- Analysis of which quantization method performs best

**Key tasks:**
1. Collect all evaluation results
2. Calculate accuracy deltas (quantized - baseline 87.19%)
3. Compare model file sizes
4. Document findings
5. Identify best-performing quantization approach

**Dependencies:**
- Phase 2 (ONNX quantized models evaluated)
- Phase 3 (PyTorch quantized models evaluated)

**Effort:** Low (data collection and reporting)

## Build Order Rationale

**Sequential dependency chain:**
```
Phase 1 (Calibration Utils)
   ├──▶ Phase 2 (ONNX Quantization)  ─┐
   │                                  │
   └──▶ Phase 3 (PyTorch Quantization)├──▶ Phase 4 (Comparison)
                                      │
                                      ┘
```

**Why this ordering:**

1. **Calibration first** - Shared dependency, enables parallel work on phases 2-3
2. **ONNX before PyTorch** - De-risk with simpler path, validate calibration data works
3. **PyTorch third** - More complex, benefit from lessons learned in ONNX quantization
4. **Comparison last** - Natural conclusion, requires all prior results

**Alternative considered:** Parallel phases 2-3 after phase 1
- **Rejected because:** PyTorch quantization can learn from ONNX quantization experience (calibration data size tuning, accuracy expectations)

## Framework-Specific Quantization Details

### ONNX Runtime

**API Location:** `onnxruntime.quantization.quantize`

**Key Functions:**
- `quant_pre_process(input_model_path, output_model_path)` - Pre-process model
- `quantize_static(model_input, model_output, calibration_data_reader, **kwargs)` - Main quantization

**Quantization Types:**
- `QuantType.QInt8` - Signed 8-bit integer
- `QuantType.QUInt8` - Unsigned 8-bit integer

**Calibration Methods:**
- MinMax (default) - Uses min/max values from calibration data
- Entropy (KL divergence) - Minimizes information loss
- Percentile - Clips outliers based on percentile

**CalibrationDataReader Interface:**
```python
class CalibrationDataReader:
    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next batch as {input_name: data} or None when done."""
        pass
```

**Output Format:** Standard ONNX file with QuantizeLinear/DequantizeLinear ops

**Confidence:** HIGH - Official ONNX Runtime documentation and examples

### PyTorch

**API Location:** `torch.ao.quantization` (legacy) or `torchao` (recommended)

**Key Functions (PT2E Path):**
- `torch.export.export(model, args)` - Export to graph representation
- `prepare_pt2e(model, quantizer)` - Insert observers
- `convert_pt2e(model)` - Convert to quantized representation

**Quantization Types:**
- `torch.qint8` - Signed 8-bit integer (standard on CPU)
- `torch.quint8` - Unsigned 8-bit integer (may have limited support)

**Backends:**
- `x86` - Default for x86 CPUs (replaced FBGEMM in PyTorch 2.x)
- `qnnpack` - Optimized for ARM/mobile

**Observer Types:**
- `MinMaxObserver` - Tracks min/max activation values
- `MovingAverageMinMaxObserver` - Smoothed min/max for stability
- `HistogramObserver` - Entropy-based calibration

**Output Format:** PyTorch checkpoint (.pt) with quantized nn.Module

**Confidence:** MEDIUM-HIGH - Official PyTorch/torchao documentation, but PT2E is newer API

**Uncertainty:** UINT8 support on x86 backend unclear - may default to INT8

## Integration Pattern: Quantize-Then-Evaluate

Both frameworks follow the same high-level pattern:

```
┌─────────────────────────────────────────┐
│  Quantization Script (One-time)         │
│  --------------------------------       │
│  1. Load FP32 model                     │
│  2. Prepare calibration data            │
│  3. Configure quantization              │
│  4. Calibrate (collect statistics)      │
│  5. Convert to quantized                │
│  6. Save quantized model                │
└─────────────────────────────────────────┘
                  │
                  │ Produces quantized artifacts
                  ▼
┌─────────────────────────────────────────┐
│  Evaluation Script (Reusable)           │
│  --------------------------------       │
│  1. Load quantized model                │
│  2. Load test data                      │
│  3. Run inference                       │
│  4. Calculate accuracy                  │
│  5. Report results                      │
└─────────────────────────────────────────┘
```

**Benefits of this pattern:**
- **Separation of concerns** - Quantization logic isolated from evaluation
- **Reusability** - Evaluation script works for FP32 and quantized models
- **Testability** - Each script independently testable
- **Consistency** - Mirrors existing convert.py → evaluate.py pattern

## Patterns to Follow

### Pattern 1: Stratified Calibration Sampling

**What:** Sample calibration data with equal representation from each class.

**Why:** Prevents quantization bias toward majority classes, ensures all classes contribute to activation statistics.

**Implementation:**
```python
def get_calibration_data(data_dir: str, num_samples: int = 200):
    images, labels, _ = load_cifar10_test(data_dir)

    # Sample 20 images per class (200 total for 10 classes)
    samples_per_class = num_samples // 10
    calibration_indices = []

    for class_idx in range(10):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        # Random sample without replacement
        sampled = np.random.choice(class_indices, samples_per_class, replace=False)
        calibration_indices.extend(sampled)

    return images[calibration_indices], labels[calibration_indices]
```

**Source:** Best practice from calibration research (128-200 samples typical)

**Confidence:** HIGH

---

### Pattern 2: Pre-Processing Before Quantization (ONNX)

**What:** Run `quant_pre_process()` before `quantize_static()`.

**Why:** Performs symbolic shape inference and model optimization (operator fusion) that preserve computation graph structure, making debugging easier.

**Implementation:**
```python
from onnxruntime.quantization import shape_inference

# Pre-process
preprocessed_model = "models/resnet8_preprocessed.onnx"
shape_inference.quant_pre_process(
    input_model_path="models/resnet8.onnx",
    output_model_path=preprocessed_model
)

# Then quantize
quantize_static(preprocessed_model, output_model, calibration_reader, ...)
```

**Source:** ONNX Runtime documentation recommendation

**Confidence:** HIGH

---

### Pattern 3: Separate Scripts Per Quantization Type

**What:** Create separate `quantize_onnx.py` and `quantize_pytorch.py` rather than unified script.

**Why:**
- APIs differ significantly between frameworks
- Configuration options framework-specific
- Easier to maintain and debug
- Mirrors existing conversion pattern (convert.py vs convert_pytorch.py)

**Confidence:** HIGH

---

### Pattern 4: Preserve Preprocessing Consistency

**What:** Use identical preprocessing in calibration and evaluation (raw pixel values [0, 255]).

**Why:** Model trained on raw pixels - normalization would invalidate calibration and cause accuracy loss.

**Critical:** Already established in v1.0 that model expects raw [0, 255] values, NOT [0, 1] normalized.

**Implementation:**
```python
# DON'T normalize for ResNet8 CIFAR-10
images = images.astype(np.float32)  # Keep [0, 255] range

# This would break quantization:
# images = images.astype(np.float32) / 255.0  # WRONG for this model
```

**Confidence:** HIGH (learned from v1.0 debugging)

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using Full Test Set for Calibration

**What:** Using all 10,000 CIFAR-10 test images for calibration.

**Why bad:**
- Unnecessary computation time (calibration runs full inference)
- Minimal accuracy improvement beyond 100-200 samples
- Risk of overfitting quantization parameters to test set

**Consequences:**
- Slow quantization process (minutes instead of seconds)
- Potential data leakage if calibration set == test set

**Instead:** Use 100-200 stratified samples from test set (separate from final evaluation) or ideally from validation set.

**Source:** Research shows 128-200 samples sufficient, diminishing returns beyond

**Confidence:** HIGH

---

### Anti-Pattern 2: Ignoring Calibration Method Selection

**What:** Using default calibration method (MinMax) without testing alternatives.

**Why bad:**
- MinMax sensitive to outliers, may choose poor quantization range
- Entropy and Percentile methods often produce better accuracy
- CNN models (like ResNet8) typically benefit from Entropy calibration

**Consequences:**
- Unnecessary accuracy loss from suboptimal quantization ranges

**Instead:**
```python
# ONNX Runtime - try Entropy first for CNNs
quantize_static(
    ...,
    calibrate_method=CalibrationMethod.Entropy  # Not just MinMax
)

# PyTorch - use HistogramObserver for entropy-based calibration
from torch.ao.quantization import HistogramObserver
qconfig = QConfig(
    activation=HistogramObserver.with_args(...),
    weight=PerChannelMinMaxObserver.with_args(...)
)
```

**Confidence:** MEDIUM-HIGH (common recommendation, but MinMax often works well enough)

---

### Anti-Pattern 3: Modifying Evaluation Scripts Unnecessarily

**What:** Rewriting evaluation logic for quantized models when existing code works.

**Why bad:**
- Code duplication
- Maintenance burden (two evaluation paths)
- Risk of introducing bugs

**Consequences:**
- Harder to compare FP32 vs quantized (different evaluation code)
- Technical debt

**Instead:**
- ONNX: Use `scripts/evaluate.py` unchanged - quantized .onnx works identically
- PyTorch: Minimal changes to model loading only, reuse evaluation logic

**Confidence:** HIGH

---

### Anti-Pattern 4: Quantizing Before Export (PyTorch)

**What:** Trying to quantize PyTorch model before `torch.export.export()` in PT2E path.

**Why bad:**
- PT2E requires exported graph representation for quantization
- Eager mode quantization (older API) has limitations and less active development
- PT2E is recommended path in PyTorch 2.x

**Consequences:**
- Incompatible with recommended quantization workflow
- May fail or produce suboptimal results

**Instead:**
```python
# CORRECT - Export first, then quantize
exported = torch.export.export(model, (example_input,))
prepared = prepare_pt2e(exported, quantizer)
# ... calibration ...
quantized = convert_pt2e(prepared)

# WRONG - Trying to quantize non-exported model
quantized = torch.quantization.quantize_static(model, ...)  # Legacy eager mode
```

**Source:** PyTorch 2.x documentation recommends PT2E over eager mode

**Confidence:** HIGH

---

### Anti-Pattern 5: Per-Tensor Quantization Only

**What:** Using per-tensor quantization when per-channel is available.

**Why bad:**
- Per-tensor uses single scale/zero-point for entire weight tensor
- Per-channel uses separate scale/zero-point per output channel
- Per-channel typically produces better accuracy (especially for Conv layers)
- Minimal inference overhead

**Consequences:**
- Unnecessary accuracy loss from coarse quantization granularity

**Instead:**
```python
# ONNX Runtime
quantize_static(..., per_channel=True)  # Enable per-channel quantization

# PyTorch
from torch.ao.quantization import PerChannelMinMaxObserver
qconfig = QConfig(
    activation=MinMaxObserver,
    weight=PerChannelMinMaxObserver  # Per-channel for weights
)
```

**Confidence:** HIGH (standard best practice)

## Verification Strategy

### Quantization Script Verification

**After quantization, verify:**
1. **Model file created** - Check file exists at expected path
2. **Model loads** - `onnxruntime.InferenceSession()` or `torch.load()` succeeds
3. **Model runs** - Forward pass with test input produces output
4. **Output shape correct** - (batch, 10) for CIFAR-10 classification
5. **Accuracy reasonable** - Run evaluation on subset (1000 images) for quick check

### Calibration Data Verification

**Before quantization, verify:**
1. **Sample count** - 200 images (or configured amount)
2. **Class distribution** - 20 images per class (stratified)
3. **Shape correct** - (N, 32, 32, 3)
4. **Value range** - [0, 255] for raw pixels (not normalized)
5. **No duplicates** - Each image appears once in calibration set

### Evaluation Verification

**After quantization, verify:**
1. **Baseline accuracy** - FP32 models still achieve 87.19% (sanity check)
2. **Quantized accuracy** - Within reasonable delta (typically -1% to -3% for 8-bit)
3. **Model size reduction** - INT8/UINT8 models ~4x smaller than FP32
4. **No degraded classes** - Per-class accuracy doesn't collapse for any class

## Risk Factors and Mitigations

### Risk 1: PyTorch UINT8 Limited Support

**Risk:** PyTorch x86 backend may not support UINT8 quantization (traditionally INT8-focused).

**Impact:** Cannot produce `resnet8_uint8.pt`, scope reduction.

**Likelihood:** Medium - UINT8 support unclear in documentation.

**Mitigation:**
1. Attempt UINT8 quantization with `torch.quint8` dtype
2. If unsupported, document limitation and proceed with INT8 only
3. ONNX Runtime UINT8 still available for comparison

**Detection:** PyTorch will raise error if UINT8 unsupported during `prepare_pt2e()`.

---

### Risk 2: Quantization Accuracy Loss > 5%

**Risk:** Quantized models lose >5% accuracy (e.g., 87.19% → <82%), making results unusable.

**Impact:** Quantized models not production-viable, investigation needed.

**Likelihood:** Low - 8-bit quantization typically loses 1-3% accuracy with proper calibration.

**Mitigation:**
1. Use entropy-based calibration (not just MinMax)
2. Increase calibration samples to 500 if accuracy poor
3. Try per-channel quantization
4. Debug with per-layer accuracy analysis

**Detection:** Evaluation script reports accuracy after quantization.

---

### Risk 3: Calibration Data Selection Bias

**Risk:** Using test set for calibration creates data leakage, overfitting quantization parameters.

**Impact:** Accuracy appears good on test set but doesn't generalize.

**Likelihood:** Low if using small subset (200/10000 = 2% overlap).

**Mitigation:**
1. Use separate validation set if available
2. Document test set usage for calibration
3. Keep calibration set small (200 images) to minimize overlap

**Detection:** Cannot detect directly (would need separate validation set).

---

### Risk 4: PT2E Export Fails for Model

**Risk:** `torch.export.export()` fails due to dynamic control flow or unsupported operations.

**Impact:** Cannot use PT2E quantization path, must fall back to eager mode.

**Likelihood:** Low - ResNet8 is simple CNN with standard operations.

**Mitigation:**
1. Test export early (Phase 3, Task 1)
2. If export fails, document issue and use eager mode quantization
3. Eager mode: `torch.quantization.quantize_static()` instead of PT2E

**Detection:** `torch.export.export()` will raise error if model not exportable.

## Confidence Assessment

| Aspect | Confidence | Rationale |
|--------|-----------|-----------|
| **ONNX Runtime workflow** | HIGH | Official documentation, clear examples, standard API |
| **PyTorch PT2E workflow** | MEDIUM-HIGH | Official documentation, but newer API (2.x), less battle-tested |
| **Calibration data requirements** | HIGH | Multiple sources agree on 100-200 samples, stratified sampling |
| **Integration points** | HIGH | Analyzed existing code, quantized models use same interfaces |
| **Build order** | HIGH | Sequential dependencies clear, de-risks with ONNX first |
| **PyTorch UINT8 support** | LOW | Documentation unclear on x86 backend UINT8 support |
| **Accuracy expectations** | MEDIUM | Typical 8-bit PTQ loses 1-3%, but model-specific |

## Open Questions for Phase Planning

1. **Calibration method comparison:** Should we try multiple calibration methods (MinMax, Entropy, Percentile) or default to Entropy?
   - **Recommendation:** Start with MinMax (simpler), add Entropy if accuracy poor

2. **Calibration sample size:** Use 200 samples (default) or more?
   - **Recommendation:** Start with 200, increase to 500 only if accuracy < 84%

3. **PyTorch backend selection:** x86 (default) or qnnpack?
   - **Recommendation:** x86 (CPU target), qnnpack only if targeting mobile/ARM

4. **Validation vs test set for calibration:** Use test set subset or separate validation?
   - **Recommendation:** Test set subset (200/10000) acceptable for research, document limitation

5. **UINT8 handling if PyTorch doesn't support:** Skip or investigate workaround?
   - **Recommendation:** Attempt first, skip if unsupported (ONNX UINT8 still available)

## Sources

### High Confidence (Official Documentation)

- [Quantize ONNX models | ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - ONNX Runtime quantization overview
- [ONNX Runtime Image Classification Quantization Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md) - Complete workflow example
- [PyTorch 2 Export Post Training Quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html) - PT2E quantization tutorial
- [Static Quantization Tutorial - PyTorch](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html) - Static quantization guide
- [INT8 Quantization for x86 CPU in PyTorch](https://pytorch.org/blog/int8-quantization/) - x86 backend details

### Medium Confidence (Research and Community)

- [Calibration Data for LLM Quantization](https://apxml.com/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/calibration-data-selection) - Calibration data best practices (128-200 samples)
- [Accurate Post Training Quantization With Small Calibration Sets](http://proceedings.mlr.press/v139/hubara21a/hubara21a.pdf) - Research on calibration set size
- [Model Quantization Concepts - NVIDIA](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) - Quantization overview
- [Quantization Overview - HuggingFace](https://huggingface.co/docs/optimum/en/concept_guides/quantization) - General quantization concepts

### Low Confidence (WebSearch, Verification Needed)

- PyTorch UINT8 support on x86 backend - Documentation unclear, needs experimental verification
- Optimal calibration method for ResNet8 CIFAR-10 - Model-specific, requires experimentation
- PT2E export compatibility with onnx2torch-converted models - May have edge cases

## Notes for Roadmap Creation

**Phase structure recommendation:**

1. **Phase 1: Calibration Infrastructure** - Foundational, enables parallel work
2. **Phase 2: ONNX Runtime Quantization** - Lower complexity, de-risk early
3. **Phase 3: PyTorch Quantization** - Higher complexity, benefits from Phase 2 learnings
4. **Phase 4: Comparison and Analysis** - Synthesize results

**Research flags:**
- **Phase 1:** Low research risk - straightforward data sampling
- **Phase 2:** Low research risk - well-documented ONNX Runtime API
- **Phase 3:** Medium research risk - PT2E is newer, UINT8 support unclear
- **Phase 4:** No research risk - data collection only

**Critical path dependencies:**
- Phase 2 and 3 both depend on Phase 1 (calibration data)
- Phase 4 depends on Phase 2 and 3 (all evaluations complete)
- No circular dependencies

**Estimated effort:**
- Phase 1: 1-2 hours (simple data sampling)
- Phase 2: 3-4 hours (CalibrationDataReader, quantization, evaluation)
- Phase 3: 4-6 hours (PT2E export, quantization, potential evaluation changes, UINT8 investigation)
- Phase 4: 1-2 hours (comparison table, documentation)

**Total estimated effort:** 9-14 hours for complete PTQ evaluation milestone.
