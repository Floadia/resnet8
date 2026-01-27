# Requirements: ResNet8 ONNX Evaluation

**Defined:** 2025-01-27
**Core Value:** Accurate Keras→ONNX conversion with >85% CIFAR-10 accuracy

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Conversion

- [ ] **CONV-01**: Convert Keras .h5 model to ONNX format using tf2onnx
- [ ] **CONV-02**: Verify ONNX model structure (input shape, output shape, layer count)
- [ ] **CONV-03**: Log conversion progress and any warnings

### Evaluation

- [ ] **EVAL-01**: Evaluate ONNX model on CIFAR-10 test set using ONNX Runtime
- [ ] **EVAL-02**: Report per-class accuracy breakdown (10 classes)
- [ ] **EVAL-03**: Achieve >85% overall accuracy on CIFAR-10 test set

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Evaluation

- **EVAL-04**: Compare with original Keras model baseline
- **EVAL-05**: Inference performance benchmarking (latency, throughput)

### PyTorch Support

- **PT-01**: Convert ONNX to PyTorch using onnx2torch
- **PT-02**: PyTorch evaluation script

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Training from scratch | Using pretrained weights only |
| Quantized model support | Full-precision evaluation only |
| TFLite conversion | ONNX is the target format |
| Custom datasets | CIFAR-10 only |
| Manual weight conversion | Using tf2onnx pipeline instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONV-01 | Phase 1 | Pending |
| CONV-02 | Phase 1 | Pending |
| CONV-03 | Phase 1 | Pending |
| EVAL-01 | Phase 2 | Pending |
| EVAL-02 | Phase 2 | Pending |
| EVAL-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 6 total
- Mapped to phases: 6
- Unmapped: 0 (100% coverage)

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-01-27 after roadmap creation*
