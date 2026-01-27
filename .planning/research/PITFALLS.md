# Pitfalls Research: Post-Training Quantization (PTQ) for ResNet8

**Domain:** Adding static PTQ to existing CNN model evaluation
**Context:** ResNet8 CIFAR-10 (87.19% full-precision baseline), ONNX Runtime + PyTorch quantization
**Researched:** 2026-01-28
**Confidence:** HIGH (Context7, official documentation, recent community discussions)

---

## Critical Pitfalls

Mistakes that cause incorrect quantized models, severe accuracy degradation, or require complete rework.

---

### Pitfall 1: Random or Insufficient Calibration Data

**What goes wrong:** Using random data for calibration or too few calibration samples produces incorrect quantization parameters (scale and zero_point), causing severe accuracy drops (20-70% degradation).

**Why it happens:**
- Code examples use random data for convenience (`torch.randn(...)`)
- Developers don't understand that calibration determines fixed quantization ranges for all future inputs
- Insufficient calibration samples fail to capture activation distribution
- Calibration dataset doesn't match inference distribution

**Consequences:**
- Quantized model accuracy drops from 87.19% to 10-50% (near-random or severely degraded)
- int8/uint8 ranges are incorrectly computed, causing clipping or poor utilization
- Quantization parameters don't generalize to test set
- PyTorch: Observers collect incorrect statistics
- ONNX Runtime: Calibration table contains wrong scale/zero_point values

**Prevention:**
1. **NEVER use random data** — always use real CIFAR-10 samples
2. Use **100+ mini-batches** for calibration (PyTorch recommendation: ~100, minimum 32)
3. Draw calibration data from training set (not test set to avoid data leakage)
4. Ensure calibration data is **representative** of inference distribution
5. For CIFAR-10: Use 1000-3200 images (100 batches × 32 batch_size, or 32 batches × 100 batch_size)
6. Apply exact same preprocessing as inference (raw pixels 0-255, no normalization)
7. Verify calibration statistics make sense (check min/max ranges per layer)

**Detection:**
- Quantized accuracy drops below 60% (expected: 85-87%)
- Activation ranges in calibration table show unusual values (all zeros, extreme outliers)
- Different calibration runs produce wildly different accuracy
- First quantized layer has scale/zero_point that clips most inputs

**Phase to address:** Calibration Data Preparation phase — create proper calibration subset before quantization

**Warning signs:**
- Calibration code uses `torch.randn()` or `np.random.randn()`
- Calibration loop iterates fewer than 32 times
- Calibration uses different preprocessing than evaluation
- No verification of calibration data distribution

---

### Pitfall 2: Model Not in Eval Mode During Calibration

**What goes wrong:** Calibrating in training mode causes BatchNorm layers to update running statistics instead of using frozen statistics, producing incorrect quantization ranges and poor accuracy.

**Why it happens:**
- Developers forget `model.eval()` is required before calibration
- Confusion between training/eval mode in PyTorch
- Copy-paste from training code where `.train()` is set
- Assuming calibration is "training the quantization"

**Consequences:**
- BatchNorm uses batch statistics during calibration instead of learned running mean/variance
- Quantization observers collect wrong activation ranges
- Quantized model accuracy degrades by 10-20%
- Results are inconsistent across calibration runs
- Small calibration batches cause catastrophic BatchNorm instability

**Prevention:**
1. **Always call `model.eval()`** before calibration:
   ```python
   model.eval()  # CRITICAL: Must be before prepare()
   model_prepared = torch.quantization.prepare(model, inplace=False)
   ```
2. Verify model mode: `assert not model.training, "Model must be in eval mode"`
3. Use `torch.no_grad()` context during calibration (prevents gradient computation)
4. Document this requirement prominently in calibration script
5. Test with single-image input — if results vary, mode is wrong

**Detection:**
- Quantized accuracy varies significantly between runs
- Single-image predictions from quantized model are unstable
- BatchNorm layers show changing statistics during calibration
- Calibration with batch_size=1 produces very different results than batch_size=32

**Phase to address:** Quantization Preparation phase — set eval mode before prepare()

**Warning signs:**
- No `model.eval()` call before `torch.quantization.prepare()`
- Calibration loop doesn't use `with torch.no_grad():`
- Model mode check not asserted

---

### Pitfall 3: Forgetting Module Fusion for Conv-BatchNorm-ReLU

**What goes wrong:** Not fusing Conv→BatchNorm→ReLU sequences before quantization causes incorrect quantization boundaries, missing optimization opportunities, and accuracy degradation.

**Why it happens:**
- Developers skip fusion step in quantization workflow
- Unfamiliarity with `fuse_modules()` requirement
- Assumption that quantization handles this automatically
- Module fusion requires explicit module names, causing confusion

**Consequences:**
- BatchNorm cannot be quantized standalone — quantization fails or skips BatchNorm
- Quantization/dequantization operations inserted between Conv-BatchNorm, causing numerical errors
- Performance is degraded (extra quant/dequant ops)
- Accuracy drops by 3-10% due to accumulated quantization errors
- ONNX Runtime: Model optimization may warn about unfused patterns

**Prevention:**
1. **Fuse modules BEFORE quantization**:
   ```python
   # Fusion must happen before prepare()
   model = torch.quantization.fuse_modules(model, [
       ['conv1', 'bn1', 'relu1'],      # Initial conv block
       ['stack1.0.conv1', 'stack1.0.bn1', 'stack1.0.relu1'],  # Residual blocks
       # ... list all Conv-BN-ReLU sequences
   ], inplace=False)
   ```
2. Fusion order must be: **Conv → BatchNorm → ReLU** (not ReLU → BatchNorm)
3. Supported fusion patterns: `[Conv, ReLU]`, `[Conv, BatchNorm]`, `[Conv, BatchNorm, ReLU]`, `[Linear, ReLU]`
4. For ResNet8: Fuse main path and shortcut projection paths separately
5. Verify fusion worked: `print(model)` should show fused modules
6. ONNX Runtime: Use `onnxruntime.quantization.shape_inference` and model optimizer before quantization

**Detection:**
- BatchNorm layers appear separately in quantized model
- Extra QuantizeLinear/DequantizeLinear pairs between Conv and BatchNorm
- Warning: "Cannot quantize BatchNormalization by itself"
- Accuracy significantly lower than expected (3-10% drop)
- Quantized model is slower than expected

**Phase to address:** Model Preparation phase — fuse modules before quantization workflow

**Warning signs:**
- No `fuse_modules()` call before `prepare()`
- Module names are not explicitly listed
- Fusion patterns don't match ResNet8 architecture

---

### Pitfall 4: Skip Connections Not Prepared for Quantization

**What goes wrong:** Residual addition operations (`out = x + residual`) fail during quantization or produce incorrect results because they aren't quantization-aware.

**Why it happens:**
- Standard Python `+` operator doesn't support quantization
- torch.nn.Identity is not inserted for activation quantization tracking
- Developers don't know to use `torch.nn.quantized.FloatFunctional`
- ResNet shortcuts seem to "just work" in floating point

**Consequences:**
- Runtime error: "Cannot add quantized and non-quantized tensors"
- No activation quantization for skip connection additions
- Erroneous quantization calibration (observers miss skip connection statistics)
- Silent failure: addition works but produces incorrect results
- Accuracy drops by 5-15% due to quantization range mismatch

**Prevention:**
1. **Replace `+` with FloatFunctional** for skip connections:
   ```python
   # WRONG: out = x + residual
   # RIGHT:
   self.skip_add = torch.nn.quantized.FloatFunctional()
   out = self.skip_add.add(x, residual)
   ```
2. Insert `torch.nn.Identity()` before additions to flag activation quantization
3. Apply this to ALL residual blocks (3 stacks × 2 blocks = 6 additions in ResNet8)
4. Test each residual block independently after quantization
5. Use QuantStub at model input and DeQuantStub at model output

**Detection:**
- Error during quantization: "Cannot quantize add operation"
- Quantized model output differs significantly from float model on same input
- Residual blocks show no observers for addition operations
- Calibration warnings about skip connections

**Phase to address:** Model Architecture Modification phase — update residual blocks before quantization

**Warning signs:**
- Residual blocks use standard `+` operator
- No `FloatFunctional` imports or instances in model code
- Model definition wasn't modified for quantization

---

### Pitfall 5: Observer Mismatch Between QConfig and Backend

**What goes wrong:** Using incompatible QConfig for the target backend (CPU, GPU, CUDA) causes silent failures, numerical saturation, or wrong inference results.

**Why it happens:**
- Default QConfig doesn't match target hardware
- OneDNN backend on non-VNNI CPUs causes saturation
- Per-tensor vs per-channel observer mismatch
- uint8 vs int8 activation dtype mismatch

**Consequences:**
- **OneDNN backend without AVX-512 VNNI**: Silent numeric saturation, model outputs are wrong
- Per-tensor quantization on efficient models (like MobileNet) causes severe accuracy drops
- Quantization config is ignored, operators remain float32
- Runtime error: "Backend does not support this quantization configuration"

**Prevention:**
1. **Choose QConfig matching backend**:
   ```python
   # CPU inference (most compatible)
   model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86
   # or
   model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # ARM
   ```
2. For ResNet8 on x86 CPU: Use `fbgemm` backend (Facebook GEMM)
3. Verify backend support: Check `torch.backends.quantized.supported_engines`
4. For OneDNN: Check CPU flags for AVX-512 VNNI support, or avoid OneDNN
5. Use per-channel quantization for weights, per-tensor for activations (ResNet standard)
6. ONNX Runtime: Verify execution provider supports quantized operators

**Detection:**
- Model outputs are saturated (all same value, NaN, or inf)
- CPU without VNNI shows wrong results but VNNI CPU is correct
- Quantized model accuracy is 0-10% (catastrophic failure)
- Logs show: "qconfig ignored for operator X"
- ONNX Runtime error: "Unsupported quantized operator for execution provider"

**Phase to address:** Quantization Configuration phase — select correct QConfig before prepare()

**Warning signs:**
- QConfig is default without backend specification
- Code doesn't check hardware capabilities
- Using OneDNN on older CPUs

---

### Pitfall 6: ONNX Runtime Quantization Without Model Optimization

**What goes wrong:** Quantizing ONNX model without pre-processing optimization causes unsupported operator errors, missed fusion opportunities, and accuracy degradation.

**Why it happens:**
- Developers skip `onnxruntime.quantization.shape_inference` step
- Model optimizer not run before quantization
- ONNX graph contains patterns that cannot be quantized
- BatchNorm not fused with Conv before quantization

**Consequences:**
- Error: "Cannot quantize BatchNormalization by itself"
- Unsupported Conv nodes cannot be quantized
- Missing tensor shape information prevents quantization
- Quantization silently skips operators, leaving them in float32
- Zero padding causes accuracy errors if zero cannot be represented uniquely

**Prevention:**
1. **Run symbolic shape inference first**:
   ```python
   from onnxruntime.quantization import shape_inference
   model_with_shapes = shape_inference.quant_pre_process(
       input_model_path='resnet8.onnx',
       output_model_path='resnet8_inferred.onnx'
   )
   ```
2. Use model optimizer to fuse Conv-BatchNorm-ReLU before quantization
3. Verify all operators have shape information
4. Check supported operators for execution provider (CPU, CUDA, TensorRT)
5. For zero padding: Ensure zero is uniquely representable in quantization scheme

**Detection:**
- Error message: "Cannot quantize BatchNormalization"
- Quantization log shows: "Unsupported operator X, skipping"
- Quantized model has mix of quantized and float32 operators
- Accuracy severely degraded (>20% drop)

**Phase to address:** ONNX Model Preparation phase — pre-process before quantization

**Warning signs:**
- No shape inference call before quantization
- ONNX model loaded directly into quantization without preprocessing
- Quantization warnings about missing shape information

---

### Pitfall 7: Incorrect Calibration Data Preprocessing

**What goes wrong:** Calibration uses different preprocessing than the original model or evaluation script, causing quantization ranges to be computed for the wrong input distribution.

**Why it happens:**
- Calibration script normalizes inputs (divides by 255, or applies mean/std normalization)
- Evaluation script uses raw pixels (0-255)
- Copy-paste from ImageNet example that uses normalization
- Misunderstanding of what the model expects

**Consequences:**
- Quantization ranges are computed for normalized data (0-1) but inference uses raw pixels (0-255)
- Activation clipping or extreme underutilization of quantization range
- Quantized accuracy drops to 10-40% (worse than random)
- First layer weights have completely wrong quantization scale

**Prevention:**
1. **Match preprocessing EXACTLY** between calibration and inference:
   ```python
   # Check evaluate_pytorch.py: images are float32 in [0, 255]
   # Calibration MUST use same format
   images = images.astype(np.float32)  # NO divide by 255
   ```
2. Review evaluation script preprocessing before writing calibration
3. Test calibration with same images used in evaluation
4. Verify input ranges: print min/max of calibration inputs
5. Document preprocessing requirements in calibration script

**Detection:**
- Quantized first Conv layer has scale >> 1 or scale << 0.01
- Calibration input stats differ from evaluation input stats
- Quantized accuracy catastrophically low (<50%)
- Activations are clipped (all at min or max quantization range)

**Phase to address:** Calibration Data Preparation phase — verify preprocessing pipeline

**Warning signs:**
- Calibration code divides by 255, but evaluate_pytorch.py doesn't
- Normalization transform applied in calibration but not evaluation
- No documentation of expected input range

---

## Moderate Pitfalls

Mistakes that cause delays, confusing errors, or require careful debugging but don't break the entire system.

---

### Pitfall 8: Calibration vs Inference Batch Size Mismatch

**What goes wrong:** Calibrating with one batch size but evaluating with a different batch size causes unexpected behavior or performance issues.

**Why it happens:**
- Calibration uses large batches (batch_size=128) for speed
- Evaluation uses small batches (batch_size=32) or single images
- Assumption that batch size doesn't matter for static quantization
- Dynamic batch dimensions not handled correctly

**Consequences:**
- Minor accuracy variations (1-2%) between calibration and evaluation batch sizes
- Runtime errors with fixed-size operators (rare for ResNet)
- Confusion during debugging when results differ
- Performance not optimized for target batch size

**Prevention:**
1. Use **same batch size** for calibration and evaluation when possible
2. For ResNet8 CIFAR-10: batch_size=32 or 100 is reasonable
3. Document batch size in calibration script
4. Test quantized model with multiple batch sizes
5. For ONNX Runtime: Use dynamic batch dimension if varying batch sizes needed

**Detection:**
- Accuracy varies slightly with batch size
- Runtime warnings about batch dimension
- Performance degrades with certain batch sizes

**Phase to address:** Calibration Configuration phase — align batch sizes

**Warning signs:**
- Calibration batch_size != evaluation batch_size
- No testing with target batch size

---

### Pitfall 9: Not Validating Quantized Model Before Comparison

**What goes wrong:** Jumping directly to accuracy comparison without validating that quantization succeeded, wasting time debugging accuracy when quantization failed silently.

**Why it happens:**
- Eagerness to see accuracy results
- Assumption that no error = success
- Lack of quantization validation steps
- Not checking for mixed float32/int8 operators

**Consequences:**
- Hours debugging accuracy when model isn't actually quantized
- Some operators remain float32, giving false sense of quantization
- Quantization silently failed but no error was raised
- Comparison is invalid (float vs partially-quantized)

**Prevention:**
1. **Validate quantization before accuracy testing**:
   ```python
   # Check model is actually quantized
   print(quantized_model)  # Should show quantized operators

   # PyTorch: Check for QuantStub/DeQuantStub
   # ONNX: Check for QuantizeLinear/DequantizeLinear nodes
   ```
2. Verify operator types: Conv should be quantized, not float
3. Check model size: int8 model should be ~4× smaller than float32
4. Run single-image sanity test: output should be close to float model (within 5%)
5. Log quantization statistics: scale, zero_point for each layer

**Detection:**
- Model size unchanged after quantization
- Print shows float32 Conv2d instead of quantized operators
- ONNX graph has no QuantizeLinear nodes
- Single-image output identical to float (not approximately equal)

**Phase to address:** Quantization Validation phase — verify before evaluation

**Warning signs:**
- No validation step between quantization and accuracy evaluation
- Model size not checked
- Operator types not inspected

---

### Pitfall 10: Comparing Quantized Accuracy Without Baseline

**What goes wrong:** Evaluating quantized model without re-running full-precision baseline, assuming 87.19% is still correct, but environment/code changes altered baseline.

**Why it happens:**
- Trusting documented baseline without verification
- Assuming evaluation script didn't change
- Not running baseline and quantized in same session
- Data loading or preprocessing changed

**Consequences:**
- Reporting incorrect accuracy delta (e.g., "quantization dropped accuracy by 5%" when baseline also dropped)
- Debugging accuracy issues that aren't related to quantization
- False conclusions about quantization quality
- Wasted effort optimizing quantization when problem is elsewhere

**Prevention:**
1. **Always run float baseline in same session**:
   ```python
   # Run both in same script
   float_acc = evaluate_model(float_model, test_data)
   quant_acc = evaluate_model(quant_model, test_data)
   delta = float_acc - quant_acc
   ```
2. Use identical data loading, preprocessing, and evaluation code
3. Document baseline accuracy with timestamp and commit hash
4. Compare against freshly-computed baseline, not historical value
5. Report delta as absolute percentage points (87.19% → 85.50% = 1.69pp drop)

**Detection:**
- Baseline accuracy doesn't match documented 87.19%
- Quantized accuracy higher than baseline (impossible)
- Debugging quantization when baseline is broken

**Phase to address:** Evaluation phase — run paired float/quantized evaluation

**Warning signs:**
- Separate scripts for float and quantized evaluation
- Baseline hardcoded as constant instead of computed
- No freshness check on baseline

---

### Pitfall 11: PyTorch vs ONNX Runtime Calibration Format Mismatch

**What goes wrong:** Preparing calibration data in PyTorch format (NCHW tensors) but ONNX Runtime expects different format or data loader interface.

**Why it happens:**
- PyTorch uses NCHW (batch, channels, height, width)
- ONNX Runtime quantization expects data reader that yields batches
- Different APIs for calibration data preparation
- Copy-paste from PyTorch example to ONNX without adaptation

**Consequences:**
- Runtime error: "Expected data reader, got tensor"
- Shape mismatch errors during calibration
- Wrong axis interpreted as channels/spatial dimensions
- Calibration fails or uses wrong data

**Prevention:**
1. **Use appropriate calibration API for each framework**:
   ```python
   # PyTorch: Direct tensor calibration
   for batch in calibration_loader:
       model(batch)

   # ONNX Runtime: DataReader class
   class CalibrationDataReader:
       def __init__(self, data):
           self.data = data
       def get_next(self):
           # Return dict: {input_name: numpy_array}
           return {'input': next_batch}
   ```
2. ONNX Runtime expects dict with input names as keys
3. PyTorch calibration uses standard forward pass
4. Verify data shapes match model input: ResNet8 expects (N, 32, 32, 3) or (N, 3, 32, 32)
5. Check channel order: ONNX may use NHWC, PyTorch uses NCHW

**Detection:**
- Type error: "Expected DataReader, got Tensor"
- Shape mismatch during calibration
- Error: "Input name not found in model"

**Phase to address:** Calibration Implementation phase — use framework-specific API

**Warning signs:**
- Reusing exact same calibration code for both frameworks
- Not checking calibration API documentation

---

### Pitfall 12: Missing QuantStub/DeQuantStub at Model Boundaries

**What goes wrong:** PyTorch quantization fails to quantize inputs/outputs properly because QuantStub and DeQuantStub are missing.

**Why it happens:**
- Tutorials don't always emphasize this requirement
- Developers expect automatic input/output quantization
- Model definition not modified for quantization
- Copy-paste of float model without stub insertion

**Consequences:**
- No quantization statistics collected for inputs
- Input remains float32, first Conv receives float instead of int8
- Output dequantization missing, returning int8 instead of float
- Runtime error: "Expected quantized tensor, got float"

**Prevention:**
1. **Add stubs to model definition**:
   ```python
   class ResNet8(nn.Module):
       def __init__(self):
           super().__init__()
           self.quant = torch.quantization.QuantStub()
           self.dequant = torch.quantization.DeQuantStub()
           # ... rest of model

       def forward(self, x):
           x = self.quant(x)  # Quantize input
           # ... model computation
           x = self.dequant(x)  # Dequantize output
           return x
   ```
2. QuantStub at model entry, DeQuantStub at model exit
3. Required for eager mode quantization in PyTorch
4. Not needed for ONNX Runtime (handles automatically)

**Detection:**
- Error: "Expected quantized input"
- Input/output quantization missing in prepared model
- First layer doesn't have quantization observer

**Phase to address:** Model Architecture Modification phase — add stubs before quantization

**Warning signs:**
- No QuantStub/DeQuantStub in model forward()
- Model definition unchanged from float version

---

## Minor Pitfalls

Mistakes that cause annoyance or confusion but are easily fixable.

---

### Pitfall 13: Quantized Model Saved Without State Dict

**What goes wrong:** Saving quantized model using `torch.save(model, path)` instead of state dict, causing loading issues or non-portability.

**Why it happens:**
- Following float model saving pattern
- Not understanding quantized model serialization requirements
- Convenience of saving entire object

**Consequences:**
- Model file includes Python code, not portable across versions
- Loading requires exact same model definition and imports
- Fails with "module not found" errors
- Larger file size than necessary

**Prevention:**
1. **Use torchscript for quantized models**:
   ```python
   # For quantized models, use TorchScript
   scripted = torch.jit.script(quantized_model)
   scripted.save('resnet8_quantized.pt')
   ```
2. Or save state dict with architecture separately
3. Document loading requirements
4. Test loading in fresh Python session

**Detection:**
- Import errors when loading model
- Model fails to load in different environment

**Phase to address:** Model Serialization phase — use correct save format

**Warning signs:**
- Using `torch.save(model, path)` for quantized model
- No TorchScript conversion

---

### Pitfall 14: Logging/Reporting Confusion Between uint8 and int8

**What goes wrong:** Results documentation says "int8" but actual quantization used uint8, causing confusion during comparison.

**Why it happens:**
- PyTorch fbgemm backend uses uint8 for activations by default
- Developers assume int8 because "8-bit quantization"
- Logging doesn't distinguish signed/unsigned
- Different frameworks use different defaults

**Consequences:**
- Confusion when comparing frameworks (PyTorch uint8 vs ONNX Runtime int8)
- Inaccurate documentation
- Wrong expectations about quantization range
- Difficult to reproduce results

**Prevention:**
1. **Log actual dtypes used**:
   ```python
   # Check and log quantization dtypes
   qconfig = model.qconfig
   print(f"Activation dtype: {qconfig.activation.dtype}")
   print(f"Weight dtype: {qconfig.weight.dtype}")
   ```
2. Document: "PyTorch fbgemm uses uint8 activations, int8 weights"
3. ONNX Runtime: Check calibration table for data types
4. Report both in results: "Quantized using uint8 (activations) and int8 (weights)"

**Detection:**
- Documentation claims int8 but model uses uint8
- Comparisons don't match expected ranges

**Phase to address:** Documentation phase — accurate reporting

**Warning signs:**
- Generic "int8 quantization" without specifying activation/weight dtypes
- No dtype verification in code

---

### Pitfall 15: Hardcoded Model Paths Break Cross-Platform Reproducibility

**What goes wrong:** Calibration or evaluation scripts hardcode paths like `/mnt/ext1/references/...`, breaking on other machines.

**Why it happens:**
- Copy-paste from evaluate_pytorch.py with hardcoded defaults
- Not using argparse for paths
- Assuming single-machine development

**Consequences:**
- Scripts fail on other machines
- Collaborators cannot run quantization
- CI/CD fails

**Prevention:**
1. **Use argparse with defaults**:
   ```python
   parser.add_argument(
       '--data-dir',
       default='/mnt/ext1/references/tiny/.../cifar-10-batches-py',
       help='Path to CIFAR-10 data'
   )
   ```
2. Support relative paths
3. Document path requirements in README
4. Check path exists, give helpful error

**Detection:**
- FileNotFoundError on other machines
- Paths fail in CI

**Phase to address:** All phases — use argparse from start

**Warning signs:**
- Absolute paths hardcoded
- No path validation

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Priority | Mitigation |
|-------|---------------|----------|------------|
| **Calibration Data Preparation** | Random/insufficient calibration data | CRITICAL | Use 1000-3200 real CIFAR-10 images, 100+ batches |
| **Calibration Data Preparation** | Preprocessing mismatch with evaluation | CRITICAL | Match evaluate_pytorch.py: raw pixels [0,255] |
| **Model Architecture Modification** | Skip connections not quantization-aware | CRITICAL | Replace `+` with FloatFunctional.add |
| **Model Architecture Modification** | Missing QuantStub/DeQuantStub | HIGH | Add stubs at model input/output |
| **Model Preparation** | Module fusion not performed | CRITICAL | Fuse Conv-BN-ReLU before prepare() |
| **Model Preparation** | Model not in eval mode | CRITICAL | Call model.eval() before prepare() |
| **Quantization Configuration** | Observer/backend mismatch | CRITICAL | Use fbgemm (x86) or qnnpack (ARM) |
| **ONNX Model Preparation** | No shape inference/optimization | HIGH | Run quant_pre_process before quantization |
| **Quantization Validation** | Skipping quantization verification | HIGH | Check model size, operator types before evaluation |
| **Evaluation** | No paired float baseline | MEDIUM | Run float and quantized in same session |
| **Evaluation** | Batch size mismatch | MEDIUM | Use consistent batch size |
| **Documentation** | uint8 vs int8 confusion | LOW | Log and report actual dtypes |

---

## Quantization Workflow Checklist

Use this checklist to avoid pitfalls when adding PTQ:

### PyTorch Static Quantization

**Pre-Quantization:**
- [ ] Model architecture modified: FloatFunctional for skip connections
- [ ] QuantStub at model input, DeQuantStub at output
- [ ] Calibration data prepared: 1000-3200 real CIFAR-10 images
- [ ] Calibration preprocessing matches evaluate_pytorch.py (raw pixels 0-255)
- [ ] QConfig selected for backend (fbgemm for x86 CPU)

**Quantization:**
- [ ] Model set to eval mode: `model.eval()`
- [ ] Modules fused: `fuse_modules()` for Conv-BN-ReLU sequences
- [ ] Model prepared: `torch.quantization.prepare(model)`
- [ ] Calibration run with `torch.no_grad()` context
- [ ] Model converted: `torch.quantization.convert(model)`

**Validation:**
- [ ] Quantized model inspected: operators show int8 types
- [ ] Model size reduced by ~4× compared to float
- [ ] Single-image test: output close to float (within 5%)
- [ ] Quantization statistics logged (scale, zero_point)

**Evaluation:**
- [ ] Float baseline computed in same session
- [ ] Quantized accuracy measured with same data/preprocessing
- [ ] Accuracy delta reported (expected: 0-3pp drop for ResNet)
- [ ] Per-class accuracy compared

### ONNX Runtime Static Quantization

**Pre-Quantization:**
- [ ] ONNX model has shape information for all tensors
- [ ] Symbolic shape inference run: `shape_inference.quant_pre_process()`
- [ ] Model optimized: Conv-BN fused
- [ ] Calibration data prepared: 1000-3200 real CIFAR-10 images
- [ ] Calibration DataReader implemented

**Quantization:**
- [ ] CalibrationDataReader yields dict with correct input names
- [ ] Calibration data in correct format (check NHWC vs NCHW)
- [ ] QuantizationMode selected (IntegerOps or QLinearOps)
- [ ] Activation type chosen (uint8 or int8)
- [ ] Quantization performed: `quantize_static()`

**Validation:**
- [ ] Quantized ONNX model has QuantizeLinear/DequantizeLinear nodes
- [ ] Model size reduced by ~4×
- [ ] Netron visualization shows int8 operators
- [ ] Calibration table saved and inspected

**Evaluation:**
- [ ] Float baseline computed with same ONNX Runtime session
- [ ] Quantized accuracy measured with same preprocessing
- [ ] Accuracy delta reported
- [ ] Per-class accuracy compared

---

## Expected Accuracy Ranges

Based on ResNet8 CIFAR-10 baseline: 87.19%

| Quantization Type | Expected Accuracy | Accuracy Drop | Concern Level |
|-------------------|-------------------|---------------|---------------|
| Float32 (baseline) | 87.19% | 0pp | — |
| int8 (well-calibrated) | 85.5-87.0% | 0-1.7pp | ✓ Good |
| int8 (decent calibration) | 83-85.5% | 1.7-4pp | ⚠ Acceptable |
| int8 (poor calibration) | 70-83% | 4-17pp | ❌ Bad — check calibration |
| int8 (broken quantization) | <70% | >17pp | ❌ Critical — quantization failed |

**Debugging guide:**
- **>85%**: Quantization successful
- **80-85%**: Check calibration data size and preprocessing
- **70-80%**: Calibration data likely wrong or insufficient
- **50-70%**: Model not in eval mode, or observer mismatch
- **<50%**: Preprocessing mismatch, or quantization completely failed

---

## Research Sources

### Critical Pitfalls (HIGH confidence)

**Calibration Data:**
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/) — "About 100 mini-batches sufficient to calibrate observers"
- [Master PyTorch Quantization: Tips, Tools, and Best Practices](https://medium.com/@noel.benji/beyond-the-basics-how-to-succeed-with-pytorch-quantization-e521ebb954cd) — "Using random data will result in bad quantization parameters"
- [Static Quantization with Eager Mode in PyTorch](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html) — Official tutorial with calibration guidance

**Module Fusion:**
- [fuse_modules — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fuse_modules.fuse_modules.html) — Official fusion API
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/) — Fusion patterns and requirements

**Skip Connections:**
- [PyTorch Static Quantization - Lei Mao's Log Book](https://leimao.github.io/blog/PyTorch-Static-Quantization/) — "Replace + with FloatFunctional.add"
- [Static Quantization — torchao 0.15 documentation](https://docs.pytorch.org/ao/stable/static_quantization.html) — QuantStub/DeQuantStub requirements

**Accuracy Degradation:**
- [Accuracy drop after model quantization - PyTorch Forums](https://discuss.pytorch.org/t/accuracy-drop-after-model-quantization/190715) — Common accuracy issues
- [Static Quantized model accuracy varies greatly with Calibration data](https://github.com/pytorch/pytorch/issues/45185) — Calibration impact on accuracy

**Observer/Backend:**
- [Default qconfig for onednn backend silently causes numeric saturation](https://github.com/pytorch/pytorch/issues/103646) — Backend compatibility issues
- [Quantization API Reference — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/quantization-support.html) — QConfig and backend configuration

**ONNX Runtime:**
- [Quantize ONNX models | ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — Official quantization guide
- [ONNX Runtime quantization README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md) — Calibration requirements

### Moderate Pitfalls (MEDIUM confidence)

- [Neural Network Quantization in PyTorch | Practical ML](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/) — Best practices
- [Quantization — PyTorch master documentation](https://glaringlee.github.io/quantization.html) — Comprehensive quantization guide
- [PyTorch to Quantized ONNX Model](https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27) — Cross-framework considerations

### Domain Knowledge (HIGH confidence)

**CNN Quantization:**
- [Quantization of Convolutional Neural Networks: Model Quantization](https://www.edge-ai-vision.com/2024/02/quantization-of-convolutional-neural-networks-model-quantization/) — CNN-specific guidance
- [Post training 4-bit quantization of convolutional networks](https://openreview.net/pdf?id=Syel64HxLS) — Accuracy degradation analysis
- [Model Quantization: Concepts, Methods, and Why It Matters | NVIDIA](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) — Quantization fundamentals

---

**Confidence Assessment:** HIGH
- Critical pitfalls verified with official documentation (PyTorch, ONNX Runtime)
- Calibration requirements confirmed across multiple authoritative sources
- Backend compatibility issues documented in PyTorch GitHub issues
- CNN-specific quantization patterns validated with recent research (2024-2026)

**Research Gaps:**
- ResNet8-specific quantization accuracy (only ResNet50 benchmarks widely available)
- CIFAR-10 quantization accuracy expectations (most research uses ImageNet)
- ONNX Runtime uint8 vs int8 trade-offs for small CNNs

These gaps should be addressed during quantization experiments by measuring actual results.
