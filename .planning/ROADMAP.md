# Roadmap: ResNet8 Model Evaluation

## Milestones

- ✅ **v1.0 ONNX Evaluation** - Phases 1-2 (shipped 2026-01-27)
- ✅ **v1.1 PyTorch Evaluation** - Phases 3-4 (shipped 2026-01-27)
- ✅ **v1.2 PTQ Evaluation** - Phases 5-8 (shipped 2026-01-28)
- ✅ **v1.3 Quantized Operations Documentation** - Phases 9-13 (shipped 2026-02-05)
- 🚧 **v1.4 Quantization Playground** - Phases 14-17 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>✅ v1.0 ONNX Evaluation (Phases 1-2) - SHIPPED 2026-01-27</summary>

### Phase 1: Model Conversion
**Goal**: ONNX model exists with verified structure matching Keras source
**Depends on**: Nothing (first phase)
**Requirements**: CONV-01, CONV-02, CONV-03
**Success Criteria** (what must be TRUE):
  1. ONNX file exists at expected path after running conversion script
  2. Conversion script logs show successful tf2onnx execution without errors
  3. ONNX model has correct input shape (1, 32, 32, 3), output shape (1, 10), and expected layer count
  4. Any conversion warnings are logged for review
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Convert Keras to ONNX and verify structure

### Phase 2: Accuracy Evaluation
**Goal**: ONNX model achieves >85% accuracy on CIFAR-10 test set
**Depends on**: Phase 1
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Evaluation script runs inference on all 10,000 CIFAR-10 test images using ONNX Runtime
  2. Per-class accuracy is reported for all 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  3. Overall accuracy is >=85% on CIFAR-10 test set
  4. Evaluation output includes total correct predictions, total samples, and percentage accuracy
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md — Evaluate ONNX model on CIFAR-10 and verify accuracy

</details>

<details>
<summary>✅ v1.1 PyTorch Evaluation (Phases 3-4) - SHIPPED 2026-01-27</summary>

### Phase 3: PyTorch Conversion
**Goal**: PyTorch model exists with verified structure matching ONNX source
**Depends on**: Phase 2 (needs ONNX model)
**Requirements**: PT-01, PT-02
**Success Criteria** (what must be TRUE):
  1. PyTorch model loads successfully from ONNX using onnx2torch
  2. Model accepts same input shape (batch, 32, 32, 3) as ONNX
  3. Model produces same output shape (batch, 10) as ONNX
  4. Conversion script logs successful onnx2torch execution
**Plans**: 1 plan

Plans:
- [x] 03-01-PLAN.md — Convert ONNX to PyTorch and verify structure

### Phase 4: PyTorch Evaluation
**Goal**: PyTorch model achieves >85% accuracy on CIFAR-10 test set
**Depends on**: Phase 3
**Requirements**: PT-03, PT-04, PT-05
**Success Criteria** (what must be TRUE):
  1. Evaluation script runs inference on all 10,000 CIFAR-10 test images using PyTorch
  2. Per-class accuracy is reported for all 10 classes
  3. Overall accuracy is >=85% on CIFAR-10 test set
  4. Evaluation output includes total correct predictions, total samples, and percentage accuracy
**Plans**: 1 plan

Plans:
- [x] 04-01-PLAN.md — Evaluate PyTorch model on CIFAR-10 and verify accuracy

</details>

<details>
<summary>✅ v1.2 PTQ Evaluation (Phases 5-8) - SHIPPED 2026-01-28</summary>

### Phase 5: Calibration Infrastructure
**Goal**: Calibration data prepared with correct preprocessing matching evaluation pipeline
**Depends on**: Phase 4 (needs evaluation infrastructure)
**Requirements**: CAL-01, CAL-02
**Success Criteria** (what must be TRUE):
  1. Calibration dataset contains 200+ stratified CIFAR-10 samples (20 per class minimum)
  2. Calibration utility script (`scripts/calibration_utils.py`) exists and loads samples correctly
  3. Calibration preprocessing exactly matches evaluation preprocessing (raw pixels 0-255, no normalization)
  4. Sample distribution verification shows balanced class representation
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md — Create calibration data loader with stratified sampling and verification

### Phase 6: ONNX Runtime Quantization
**Goal**: ONNX models quantized to int8/uint8 with evaluated accuracy vs baseline
**Depends on**: Phase 5
**Requirements**: ORT-01, ORT-02, ORT-03, ORT-04
**Success Criteria** (what must be TRUE):
  1. Quantized ONNX models exist (resnet8_int8.onnx and resnet8_uint8.onnx)
  2. Both quantized models evaluate successfully on CIFAR-10 test set using existing evaluation script
  3. Accuracy delta reported for int8 model vs 87.19% baseline
  4. Accuracy delta reported for uint8 model vs 87.19% baseline
  5. Quantization script logs calibration method used (MinMax) and sample count
**Plans**: 1 plan

Plans:
- [x] 06-01-PLAN.md — Quantize ONNX model to int8/uint8 and evaluate accuracy

### Phase 7: PyTorch Quantization
**Goal**: PyTorch models quantized to int8/uint8 with evaluated accuracy vs baseline
**Depends on**: Phase 6 (benefits from ONNX lessons learned)
**Requirements**: PTQ-01, PTQ-02, PTQ-03, PTQ-04
**Success Criteria** (what must be TRUE):
  1. Quantized PyTorch model exists (resnet8_int8.pt)
  2. uint8 model exists if fbgemm backend supports it, otherwise documented as unsupported
  3. Quantized models evaluate successfully on CIFAR-10 test set
  4. Accuracy delta reported for int8 model vs 87.19% baseline
  5. Accuracy delta reported for uint8 model (if created) vs 87.19% baseline
**Plans**: 1 plan

Plans:
- [x] 07-01-PLAN.md — Quantize PyTorch model to int8 and evaluate accuracy

### Phase 8: Comparison and Analysis
**Goal**: All quantization results compared with accuracy deltas flagged
**Depends on**: Phases 6 and 7
**Requirements**: ANL-01, ANL-02
**Success Criteria** (what must be TRUE):
  1. Comparison table exists showing: Framework × Data Type × Accuracy × Delta from baseline
  2. All configurations with accuracy drop >5% are flagged in analysis
  3. Model size comparison included (FP32 vs quantized for all models)
  4. Analysis document includes recommendation for best quantization approach
**Plans**: 1 plan

Plans:
- [x] 08-01-PLAN.md — Create quantization comparison analysis document

</details>

<details>
<summary>✅ v1.3 Quantized Operations Documentation (Phases 9-13) - SHIPPED 2026-02-05</summary>

### Phase 9: Operation Extraction Scripts
**Goal**: Programmatic tools extract quantized operation details from ONNX models for data-driven documentation
**Depends on**: Phase 8 (needs quantized ONNX models from v1.2)
**Requirements**: TOOL-01, TOOL-02
**Success Criteria** (what must be TRUE):
  1. Extraction script outputs JSON containing all QLinear nodes with their scales, zero-points, and attributes from resnet8_int8.onnx
  2. Extraction script identifies all quantized operation types (QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear)
  3. Visualization script generates PNG/SVG graph diagrams of quantized ResNet8 model using onnx.tools.net_drawer
  4. Generated visualizations clearly show operator types and data flow from input to output
**Plans**: 1 plan

Plans:
- [x] 09-01-PLAN.md — Create extraction and visualization scripts for quantized ONNX models

### Phase 10: Boundary Operations Documentation
**Goal**: QuantizeLinear and DequantizeLinear operations fully documented with formulas and hardware guidance
**Depends on**: Phase 9 (needs extracted operation parameters)
**Requirements**: BOUND-01, BOUND-02
**Success Criteria** (what must be TRUE):
  1. QuantizeLinear documentation includes exact formula (q = saturate(round(x/scale) + zero_point)), numerical example, and hardware pseudocode
  2. DequantizeLinear documentation includes exact formula (x = (q - zero_point) × scale), numerical example, and hardware pseudocode
  3. Documentation renders correctly on GitHub with LaTeX math equations (MathJax support validated)
  4. Boundary operations documentation explains FP32-to-INT8 and INT8-to-FP32 conversions at model input/output
**Plans**: 1 plan

Plans:
- [x] 10-01-PLAN.md — Document QuantizeLinear and DequantizeLinear operations with formulas and examples

### Phase 11: Core Operations Documentation
**Goal**: QLinearConv and QLinearMatMul operations fully documented with two-stage computation explained
**Depends on**: Phase 10 (builds on boundary operations foundation)
**Requirements**: CORE-01, CORE-02, CORE-03
**Success Criteria** (what must be TRUE):
  1. QLinearConv documentation covers all 9 inputs (x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, bias) and explains two-stage computation (INT8×INT8→INT32 MAC, then requantization to INT8)
  2. QLinearMatMul documentation covers input structure, computation stages, and INT32 accumulator requirements
  3. Per-channel quantization handling documented (scale[c] per output channel vs scalar scale, memory requirements)
  4. Worked examples use actual ResNet8 layer values (Conv2D(16, 3×3) or similar) showing all intermediate calculations with exact bit-widths
  5. Hardware implementation pseudocode specifies INT32 accumulator requirement to prevent overflow
**Plans**: 2 plans

Plans:
- [x] 11-01-PLAN.md — QLinearConv documentation with two-stage computation and validation
- [x] 11-02-PLAN.md — QLinearMatMul documentation for FC layer

### Phase 12: Architecture Documentation
**Goal**: Full ResNet8 quantized architecture documented with scale/zero-point flow and residual connection handling
**Depends on**: Phase 11 (needs core operations context)
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. Data flow diagram shows complete path through quantized ResNet8 (FP32 input → QuantizeLinear → INT8 layers → DequantizeLinear → FP32 output)
  2. Scale and zero-point parameter locations documented in ONNX graph (which initializers, where they appear in operator inputs)
  3. Residual connection handling documented with scale mismatch problem explained and solution approaches compared (QDQ dequant-add-quant, scale matching, PyTorch FloatFunctional)
  4. PyTorch quantized operation equivalents mapped to ONNX operations (e.g., torch.nn.quantized.Conv2d → QLinearConv)
  5. Full network visualization included showing all quantized operations with annotated scale/zero-point flow
**Plans**: 2 plans

Plans:
- [x] 12-01-PLAN.md — QDQ architecture documentation with network visualization and scale/zero-point locations
- [x] 12-02-PLAN.md — Residual connections and PyTorch equivalents documentation

### Phase 13: Hardware Implementation Guide
**Goal**: Complete hardware implementation checklist with critical pitfalls, pseudocode, and test vectors
**Depends on**: Phase 12 (needs full architecture context)
**Requirements**: HW-01, HW-02, HW-03
**Success Criteria** (what must be TRUE):
  1. Critical pitfalls checklist covers all 6 items: INT32 accumulator overflow prevention, round-to-nearest-even rounding mode, float32 scale precision, per-channel quantization indexing, operation fusion (Conv-ReLU), and INT8 saturation/clipping
  2. Hardware pseudocode (C-style) shows exact bit-widths for each operation stage (8×8→32 MAC, 32→8 requantization)
  3. Pseudocode includes correct rounding logic: (value >= 0) ? (value + 0.5) >> frac_bits : (value - 0.5) >> frac_bits
  4. Verification test vectors extracted from ResNet8 actual layer outputs (input tensor, weights, scales, zero-points, expected output) for hardware validation
  5. Each critical pitfall includes example showing what breaks if not handled correctly
**Plans**: 1 plan

Plans:
- [x] 13-01-PLAN.md — Create hardware implementation guide with pitfalls, pseudocode, and test vectors

</details>

### 🚧 v1.4 Quantization Playground (In Progress)

**Milestone Goal:** Interactive Marimo notebook for inspecting and experimenting with quantization parameters, enabling users to understand how scale/zero-point choices affect model accuracy.

#### Phase 14: Notebook Foundation
**Goal**: Users can launch Marimo notebook and load quantized models with proper caching for interactive experimentation
**Depends on**: Phase 13 (needs quantized models and extraction tools from v1.2-v1.3)
**Requirements**: NB-01, NB-02, NB-03, NB-04
**Success Criteria** (what must be TRUE):
  1. User can run `marimo edit playground/quantization.py` and see the notebook interface
  2. User can load ONNX quantized model (resnet8_int8.onnx) without memory leak on repeated cell execution
  3. User can load PyTorch quantized model (resnet8_int8.pt) without memory leak on repeated cell execution
  4. User can select a layer/operation from a dropdown populated with model structure
**Plans**: 2 plans

Plans:
- [x] 14-01-PLAN.md — Notebook skeleton with cached model loading utilities
- [x] 14-02-PLAN.md — Layer inspection utilities and complete UI wiring

#### Phase 15: Parameter Inspection
**Goal**: Users can explore all quantization parameters (scales, zero-points, weights) with comparison to FP32 values
**Depends on**: Phase 14 (needs model loading infrastructure)
**Requirements**: INSP-01, INSP-02, INSP-03, INSP-04, INSP-05
**Success Criteria** (what must be TRUE):
  1. User can view scale and zero-point values for any selected layer in a formatted table
  2. User can view weight tensor shapes and dtypes (INT8 vs FP32) for selected layer
  3. User can navigate full model structure via tree or list view and drill into any layer
  4. User can see FP32 vs quantized weight values side-by-side for selected layer
  5. User can view activation histogram showing distribution of values per layer
**Plans**: 2 plans

Plans:
- [x] 15-01-PLAN.md — Parameter extraction utilities and enhanced layer dropdown
- [x] 15-02-PLAN.md — Heatmap overview, parameter table, and weight histograms

#### Phase 16: Inference and Value Capture
**Goal**: Users can run inference and capture intermediate activations to understand quantization effects at each layer
**Depends on**: Phase 15 (needs parameter inspection to contextualize captured values)
**Requirements**: CAP-01, CAP-02, CAP-03, CAP-04
**Success Criteria** (what must be TRUE):
  1. User can select a CIFAR-10 sample image and run inference through the quantized model
  2. User can capture and view intermediate activation tensors at any layer during inference
  3. User can see SQNR (Signal-to-Quantization-Noise Ratio) metric for each layer comparing quantized vs FP32 activations
  4. User can see per-layer accuracy contribution analysis showing which layers degrade accuracy most
**Plans**: TBD

Plans:
- [ ] TBD

#### Phase 17: Interactive Modification
**Goal**: Users can modify quantization parameters and immediately observe the effect on inference outputs
**Depends on**: Phase 16 (needs inference and value capture to compare before/after)
**Requirements**: MOD-01, MOD-02, MOD-03
**Success Criteria** (what must be TRUE):
  1. User can modify scale and/or zero-point values for a selected layer via slider or input field
  2. User can trigger re-inference with modified parameters and see updated outputs
  3. User can compare original vs modified outputs (activations, final predictions) side-by-side
**Plans**: TBD

Plans:
- [ ] TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Model Conversion | v1.0 | 1/1 | Complete | 2026-01-27 |
| 2. Accuracy Evaluation | v1.0 | 1/1 | Complete | 2026-01-27 |
| 3. PyTorch Conversion | v1.1 | 1/1 | Complete | 2026-01-27 |
| 4. PyTorch Evaluation | v1.1 | 1/1 | Complete | 2026-01-27 |
| 5. Calibration Infrastructure | v1.2 | 1/1 | Complete | 2026-01-28 |
| 6. ONNX Runtime Quantization | v1.2 | 1/1 | Complete | 2026-01-28 |
| 7. PyTorch Quantization | v1.2 | 1/1 | Complete | 2026-01-28 |
| 8. Comparison and Analysis | v1.2 | 1/1 | Complete | 2026-01-28 |
| 9. Operation Extraction Scripts | v1.3 | 1/1 | Complete | 2026-02-02 |
| 10. Boundary Operations Documentation | v1.3 | 1/1 | Complete | 2026-02-02 |
| 11. Core Operations Documentation | v1.3 | 2/2 | Complete | 2026-02-03 |
| 12. Architecture Documentation | v1.3 | 2/2 | Complete | 2026-02-03 |
| 13. Hardware Implementation Guide | v1.3 | 1/1 | Complete | 2026-02-05 |
| 14. Notebook Foundation | v1.4 | 2/2 | Complete | 2026-02-06 |
| 15. Parameter Inspection | v1.4 | 2/2 | Complete | 2026-02-07 |
| 16. Inference and Value Capture | v1.4 | 0/TBD | Not started | - |
| 17. Interactive Modification | v1.4 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-02-07 with Phase 15 complete*
