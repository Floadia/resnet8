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

## v1.2 Requirements (Complete)

### Calibration

- [x] **CAL-01**: Calibration dataset prepared from CIFAR-10 (1000+ stratified samples)
- [x] **CAL-02**: Calibration preprocessing matches evaluation pipeline exactly

### ONNX Runtime Quantization

- [x] **ORT-01**: ONNX model quantized to int8 using static quantization
- [x] **ORT-02**: ONNX model quantized to uint8 using static quantization
- [x] **ORT-03**: Quantized ONNX models evaluated on CIFAR-10 test set
- [x] **ORT-04**: Accuracy delta reported vs 87.19% baseline

### PyTorch Quantization

- [x] **PTQ-01**: PyTorch model quantized to int8 using static quantization
- [x] **PTQ-02**: PyTorch model quantized to uint8 — documented as not supported (fbgemm limitation)
- [x] **PTQ-03**: Quantized PyTorch models evaluated on CIFAR-10 test set
- [x] **PTQ-04**: Accuracy delta reported vs 87.19% baseline

### Analysis

- [x] **ANL-01**: Comparison table showing all quantization results (framework × dtype)
- [x] **ANL-02**: Flag any configuration with accuracy drop >5%

## v1.3 Requirements (Current Milestone)

### Extraction & Visualization Tools

- [ ] **TOOL-01**: Script extracts all QLinear nodes from ONNX models with scales, zero-points, and attributes as JSON
- [ ] **TOOL-02**: Script generates PNG/SVG graph visualizations of quantized ONNX models

### Boundary Operations Documentation

- [ ] **BOUND-01**: QuantizeLinear operation documented with exact formula, numerical example, and hardware pseudocode
- [ ] **BOUND-02**: DequantizeLinear operation documented with exact formula, numerical example, and hardware pseudocode

### Core Operations Documentation

- [ ] **CORE-01**: QLinearConv documented with all 9 inputs, two-stage computation (MAC + requantization), per-channel handling
- [ ] **CORE-02**: QLinearMatMul documented with inputs, computation stages, and hardware requirements
- [ ] **CORE-03**: Worked examples using actual ResNet8 layer values with intermediate calculations

### Architecture Documentation

- [ ] **ARCH-01**: Data flow diagram through quantized ResNet8 (FP32 input → INT8 → FP32 output)
- [ ] **ARCH-02**: Scale/zero-point parameter locations documented (where they appear in ONNX graph)
- [ ] **ARCH-03**: Residual connection handling documented (scale mismatch at Add operations)
- [ ] **ARCH-04**: PyTorch quantized operation equivalents mapped to ONNX operations

### Hardware Implementation Guide

- [ ] **HW-01**: Critical pitfalls checklist with 6 items (accumulator overflow, rounding, scale precision, per-channel, fusion, clipping)
- [ ] **HW-02**: Hardware pseudocode (C-style) with exact bit-widths for each operation
- [ ] **HW-03**: Verification test vectors from ResNet8 for hardware validation

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Evaluation

- **EVAL-04**: Compare with original Keras model baseline
- **EVAL-05**: Inference performance benchmarking (latency, throughput)

### Advanced Quantization

- **ADV-01**: Quantization-aware training (QAT) for better accuracy
- **ADV-02**: Per-channel quantization exploration
- **ADV-03**: Mixed precision quantization (INT4 weights, INT8 activations)

### Advanced Documentation

- **DOC-01**: Dynamic quantization documentation
- **DOC-02**: Calibration methodology deep-dive
- **DOC-03**: Performance profiling documentation (MACs/FLOPs)

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
| Verilog/VHDL implementation | Documentation focuses on algorithms, not HDL |

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
| CAL-01 | Phase 5 | Complete |
| CAL-02 | Phase 5 | Complete |
| ORT-01 | Phase 6 | Complete |
| ORT-02 | Phase 6 | Complete |
| ORT-03 | Phase 6 | Complete |
| ORT-04 | Phase 6 | Complete |
| PTQ-01 | Phase 7 | Complete |
| PTQ-02 | Phase 7 | Complete |
| PTQ-03 | Phase 7 | Complete |
| PTQ-04 | Phase 7 | Complete |
| ANL-01 | Phase 8 | Complete |
| ANL-02 | Phase 8 | Complete |
| TOOL-01 | Phase 9 | Pending |
| TOOL-02 | Phase 9 | Pending |
| BOUND-01 | Phase 10 | Pending |
| BOUND-02 | Phase 10 | Pending |
| CORE-01 | Phase 11 | Pending |
| CORE-02 | Phase 11 | Pending |
| CORE-03 | Phase 11 | Pending |
| ARCH-01 | Phase 12 | Pending |
| ARCH-02 | Phase 12 | Pending |
| ARCH-03 | Phase 12 | Pending |
| ARCH-04 | Phase 12 | Pending |
| HW-01 | Phase 13 | Pending |
| HW-02 | Phase 13 | Pending |
| HW-03 | Phase 13 | Pending |

**Coverage:**
- v1.0 requirements: 6 total (Complete)
- v1.1 requirements: 5 total (Complete)
- v1.2 requirements: 12 total (Complete)
- v1.3 requirements: 14 total (Pending)
- Mapped to phases: 37/37 (100% coverage)

---
*Requirements defined: 2025-01-27*
*Last updated: 2026-02-02 with v1.3 traceability added*
