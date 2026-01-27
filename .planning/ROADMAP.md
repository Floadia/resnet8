# Roadmap: ResNet8 Model Evaluation

## Milestones

- ✅ **v1.0 ONNX Evaluation** - Phases 1-2 (shipped 2026-01-27)
- ✅ **v1.1 PyTorch Evaluation** - Phases 3-4 (shipped 2026-01-27)
- 🚧 **v1.2 PTQ Evaluation** - Phases 5-8 (in progress)

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

### 🚧 v1.2 PTQ Evaluation (In Progress)

**Milestone Goal:** Apply Post-Training Quantization (static) using both ONNX Runtime and PyTorch, evaluate int8 and uint8 model accuracy against full-precision baseline (87.19%)

#### Phase 5: Calibration Infrastructure
**Goal**: Calibration data prepared with correct preprocessing matching evaluation pipeline
**Depends on**: Phase 4 (needs evaluation infrastructure)
**Requirements**: CAL-01, CAL-02
**Success Criteria** (what must be TRUE):
  1. Calibration dataset contains 200+ stratified CIFAR-10 samples (20 per class minimum)
  2. Calibration utility script (`scripts/calibration_utils.py`) exists and loads samples correctly
  3. Calibration preprocessing exactly matches evaluation preprocessing (raw pixels 0-255, no normalization)
  4. Sample distribution verification shows balanced class representation
**Plans**: TBD

Plans:
- [ ] TBD

#### Phase 6: ONNX Runtime Quantization
**Goal**: ONNX models quantized to int8/uint8 with evaluated accuracy vs baseline
**Depends on**: Phase 5
**Requirements**: ORT-01, ORT-02, ORT-03, ORT-04
**Success Criteria** (what must be TRUE):
  1. Quantized ONNX models exist (resnet8_int8.onnx and resnet8_uint8.onnx)
  2. Both quantized models evaluate successfully on CIFAR-10 test set using existing evaluation script
  3. Accuracy delta reported for int8 model vs 87.19% baseline
  4. Accuracy delta reported for uint8 model vs 87.19% baseline
  5. Quantization script logs calibration method used (MinMax) and sample count
**Plans**: TBD

Plans:
- [ ] TBD

#### Phase 7: PyTorch Quantization
**Goal**: PyTorch models quantized to int8/uint8 with evaluated accuracy vs baseline
**Depends on**: Phase 6 (benefits from ONNX lessons learned)
**Requirements**: PTQ-01, PTQ-02, PTQ-03, PTQ-04
**Success Criteria** (what must be TRUE):
  1. Quantized PyTorch model exists (resnet8_int8.pt)
  2. uint8 model exists if fbgemm backend supports it, otherwise documented as unsupported
  3. Quantized models evaluate successfully on CIFAR-10 test set
  4. Accuracy delta reported for int8 model vs 87.19% baseline
  5. Accuracy delta reported for uint8 model (if created) vs 87.19% baseline
**Plans**: TBD

Plans:
- [ ] TBD

#### Phase 8: Comparison and Analysis
**Goal**: All quantization results compared with accuracy deltas flagged
**Depends on**: Phases 6 and 7
**Requirements**: ANL-01, ANL-02
**Success Criteria** (what must be TRUE):
  1. Comparison table exists showing: Framework × Data Type × Accuracy × Delta from baseline
  2. All configurations with accuracy drop >5% are flagged in analysis
  3. Model size comparison included (FP32 vs quantized for all models)
  4. Analysis document includes recommendation for best quantization approach
**Plans**: TBD

Plans:
- [ ] TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Model Conversion | v1.0 | 1/1 | Complete | 2026-01-27 |
| 2. Accuracy Evaluation | v1.0 | 1/1 | Complete | 2026-01-27 |
| 3. PyTorch Conversion | v1.1 | 1/1 | Complete | 2026-01-27 |
| 4. PyTorch Evaluation | v1.1 | 1/1 | Complete | 2026-01-27 |
| 5. Calibration Infrastructure | v1.2 | 0/TBD | Not started | - |
| 6. ONNX Runtime Quantization | v1.2 | 0/TBD | Not started | - |
| 7. PyTorch Quantization | v1.2 | 0/TBD | Not started | - |
| 8. Comparison and Analysis | v1.2 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-01-28 with v1.2 PTQ Evaluation phases*
