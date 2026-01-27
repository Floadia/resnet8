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

## v1.1 Requirements (Complete)

### PyTorch Conversion

- [x] **PT-01**: Convert ONNX model to PyTorch using onnx2torch
- [x] **PT-02**: Verify PyTorch model structure matches ONNX source

### PyTorch Evaluation

- [x] **PT-03**: Evaluate PyTorch model on CIFAR-10 test set
- [x] **PT-04**: Report per-class accuracy breakdown (10 classes)
- [x] **PT-05**: Achieve >85% overall accuracy on CIFAR-10 test set

## v1.2 Requirements (Current Milestone)

### Calibration

- [ ] **CAL-01**: Calibration dataset prepared from CIFAR-10 (1000+ stratified samples)
- [ ] **CAL-02**: Calibration preprocessing matches evaluation pipeline exactly

### ONNX Runtime Quantization

- [ ] **ORT-01**: ONNX model quantized to int8 using static quantization
- [ ] **ORT-02**: ONNX model quantized to uint8 using static quantization
- [ ] **ORT-03**: Quantized ONNX models evaluated on CIFAR-10 test set
- [ ] **ORT-04**: Accuracy delta reported vs 87.19% baseline

### PyTorch Quantization

- [ ] **PTQ-01**: PyTorch model quantized to int8 using static quantization
- [ ] **PTQ-02**: PyTorch model quantized to uint8 using static quantization (if supported)
- [ ] **PTQ-03**: Quantized PyTorch models evaluated on CIFAR-10 test set
- [ ] **PTQ-04**: Accuracy delta reported vs 87.19% baseline

### Analysis

- [ ] **ANL-01**: Comparison table showing all quantization results (framework × dtype)
- [ ] **ANL-02**: Flag any configuration with accuracy drop >5%

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Evaluation

- **EVAL-04**: Compare with original Keras model baseline
- **EVAL-05**: Inference performance benchmarking (latency, throughput)

### Advanced Quantization

- **ADV-01**: Quantization-aware training (QAT) for better accuracy
- **ADV-02**: Per-channel quantization exploration
- **ADV-03**: Mixed precision quantization

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Training from scratch | Using pretrained weights only |
| Dynamic quantization | Static quantization only (user requested) |
| TFLite quantization | Focus on ONNX Runtime and PyTorch only |
| Quantization-aware training | PTQ only — QAT deferred to future |
| Inference speed benchmarking | Accuracy only for this milestone |
| Model deployment | Evaluation only, not deployment artifacts |

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
| PT-03 | Phase 4 | Complete |
| PT-04 | Phase 4 | Complete |
| PT-05 | Phase 4 | Complete |
| CAL-01 | Phase 5 | Pending |
| CAL-02 | Phase 5 | Pending |
| ORT-01 | Phase 6 | Pending |
| ORT-02 | Phase 6 | Pending |
| ORT-03 | Phase 6 | Pending |
| ORT-04 | Phase 6 | Pending |
| PTQ-01 | Phase 7 | Pending |
| PTQ-02 | Phase 7 | Pending |
| PTQ-03 | Phase 7 | Pending |
| PTQ-04 | Phase 7 | Pending |
| ANL-01 | Phase 8 | Pending |
| ANL-02 | Phase 8 | Pending |

**Coverage:**
- v1.0 requirements: 6 total (Complete)
- v1.1 requirements: 5 total (Complete)
- v1.2 requirements: 12 total
- Mapped to phases: 12/12 ✓
- Unmapped: 0

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-01-28 with v1.2 phase mappings*
