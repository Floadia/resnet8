# Roadmap: ResNet8 Model Evaluation

## Overview

Multi-framework evaluation of ResNet8 for CIFAR-10. v1.0 completed ONNX conversion and evaluation (87.19% accuracy). v1.1 adds PyTorch conversion and evaluation using onnx2torch, targeting same >85% accuracy.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

### v1.0 (Complete)
- [x] **Phase 1: Model Conversion** - Convert Keras .h5 to ONNX and verify structure
- [x] **Phase 2: Accuracy Evaluation** - Evaluate ONNX model on CIFAR-10 test set

### v1.1 (Current)
- [x] **Phase 3: PyTorch Conversion** - Convert ONNX to PyTorch and verify structure
- [ ] **Phase 4: PyTorch Evaluation** - Evaluate PyTorch model on CIFAR-10 test set

## Phase Details

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
- [ ] 04-01-PLAN.md — Evaluate PyTorch model on CIFAR-10 and verify accuracy

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Model Conversion | 1/1 | Complete | 2026-01-27 |
| 2. Accuracy Evaluation | 1/1 | Complete | 2026-01-27 |
| 3. PyTorch Conversion | 1/1 | Complete | 2026-01-27 |
| 4. PyTorch Evaluation | 0/1 | Not started | - |

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-01-27 after v1.1 milestone start*
