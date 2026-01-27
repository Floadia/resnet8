# Roadmap: ResNet8 ONNX Evaluation

## Overview

Convert pretrained Keras ResNet8 model to ONNX format and validate accuracy on CIFAR-10 test set. Phase 1 handles conversion and verification of model structure. Phase 2 evaluates the converted model against CIFAR-10, validating >85% accuracy target.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Model Conversion** - Convert Keras .h5 to ONNX and verify structure
- [ ] **Phase 2: Accuracy Evaluation** - Evaluate ONNX model on CIFAR-10 test set

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
**Plans**: TBD

Plans:
- (Plans will be added during plan-phase)

### Phase 2: Accuracy Evaluation
**Goal**: ONNX model achieves >85% accuracy on CIFAR-10 test set
**Depends on**: Phase 1
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Evaluation script runs inference on all 10,000 CIFAR-10 test images using ONNX Runtime
  2. Per-class accuracy is reported for all 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  3. Overall accuracy is ≥85% on CIFAR-10 test set
  4. Evaluation output includes total correct predictions, total samples, and percentage accuracy
**Plans**: TBD

Plans:
- (Plans will be added during plan-phase)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Model Conversion | 0/? | Not started | - |
| 2. Accuracy Evaluation | 0/? | Not started | - |

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-01-27*
