# Requirements: ResNet8 Model Evaluation

**Defined:** 2025-01-27
**Core Value:** Accurate model conversion across frameworks with >85% CIFAR-10 accuracy

## v1.0 Requirements (Complete)

### Conversion (Keras → ONNX)

- [x] **CONV-01**: Convert Keras .h5 model to ONNX format using tf2onnx
- [x] **CONV-02**: Verify ONNX model structure (input shape, output shape, layer count)
- [x] **CONV-03**: Log conversion progress and any warnings

### ONNX Evaluation

- [x] **EVAL-01**: Evaluate ONNX model on CIFAR-10 test set using ONNX Runtime
- [x] **EVAL-02**: Report per-class accuracy breakdown (10 classes)
- [x] **EVAL-03**: Achieve >85% overall accuracy on CIFAR-10 test set

## v1.1 Requirements (Current Milestone)

### PyTorch Conversion

- [x] **PT-01**: Convert ONNX model to PyTorch using onnx2torch
- [x] **PT-02**: Verify PyTorch model structure matches ONNX source

### PyTorch Evaluation

- [ ] **PT-03**: Evaluate PyTorch model on CIFAR-10 test set
- [ ] **PT-04**: Report per-class accuracy breakdown (10 classes)
- [ ] **PT-05**: Achieve >85% overall accuracy on CIFAR-10 test set

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Evaluation

- **EVAL-04**: Compare with original Keras model baseline
- **EVAL-05**: Inference performance benchmarking (latency, throughput)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Training from scratch | Using pretrained weights only |
| Quantized model support | Full-precision evaluation only |
| TFLite conversion | ONNX is the target format |
| Custom datasets | CIFAR-10 only |
| Manual weight transfer | Using onnx2torch instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONV-01 | Phase 1 | Complete |
| CONV-02 | Phase 1 | Complete |
| CONV-03 | Phase 1 | Complete |
| EVAL-01 | Phase 2 | Complete |
| EVAL-02 | Phase 2 | Complete |
| EVAL-03 | Phase 2 | Complete |
| PT-01 | Phase 3 | Complete |
| PT-02 | Phase 3 | Complete |
| PT-03 | Phase 4 | Pending |
| PT-04 | Phase 4 | Pending |
| PT-05 | Phase 4 | Pending |

**Coverage:**
- v1.0 requirements: 6 total (Complete)
- v1.1 requirements: 5 total
- Mapped to phases: 5
- Unmapped: 0 (100% coverage)

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-01-27 after v1.1 milestone start*
