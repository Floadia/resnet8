# Project Research Summary

**Project:** ResNet8 CIFAR-10 Model Conversion (Keras to PyTorch)
**Domain:** Deep learning model conversion
**Researched:** 2026-01-27
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project converts a pretrained Keras ResNet8 model (.h5) to PyTorch and validates it against CIFAR-10 test data (target: ≥85% accuracy). Based on comprehensive research, the recommended approach is **manual weight conversion** using h5py for weight extraction and layer-by-layer mapping, rather than automated tools or ONNX intermediates.

The conversion requires careful handling of framework differences: Keras uses NHWC (channels-last) data format while PyTorch uses NCHW (channels-first), requiring weight tensor transpositions. Conv2D weights must be transposed from (H,W,C_in,C_out) to (C_out,C_in,H,W), and Dense/Linear weights from (out,in) to (in,out). BatchNorm parameters need careful mapping with epsilon consistency (Keras default: 1e-3 vs PyTorch default: 1e-5).

The primary risks are channel order mismatches causing silent failures, BatchNorm mode confusion leading to accuracy drops, and residual connection dimension mismatches in stride=2 blocks. These are mitigated through validation-driven conversion with numerical verification at each stage and comprehensive testing before full evaluation.

## Key Findings

### Recommended Stack

**Core framework:** PyTorch 2.10.0 (latest stable, 2025) with torchvision 0.25 provides the evaluation infrastructure. Python ≥3.10 is required by all dependencies. The conversion strategy prioritizes manual weight mapping over automated tools for transparency and control.

**Core technologies:**
- **PyTorch 2.10.0**: Deep learning framework — latest stable release with NumPy 2.x compatibility
- **torchvision 0.25**: Vision utilities and CIFAR-10 loader — official PyTorch companion library
- **h5py 3.15.1**: HDF5 file reader — direct access to Keras .h5 weight files without TensorFlow dependency
- **numpy**: Array operations — compatible with both PyTorch 2.10 and h5py 3.15.1
- **torchmetrics 1.8.2**: Accuracy metrics — standardized evaluation tools
- **pytest**: Testing framework — simpler than unittest for validation tests

**Fallback conversion path (if needed):**
- **tf2onnx 1.16.1**: Keras to ONNX converter
- **onnx 1.20.1**: ONNX format support
- **onnx2torch 1.5.15**: ONNX to PyTorch converter (NOT onnx2pytorch — inactive since 2021)

**Rationale for manual approach:** ResNet8 has a simple, well-understood architecture (8 layers, standard conv/bn/relu blocks). Manual conversion provides full transparency, easier debugging, and no black-box tool dependencies. ONNX path adds complexity and potential operator support issues.

### Expected Features

**Must have (table stakes):**
- **Architecture reconstruction** — PyTorch model structure matching Keras exactly (Conv2D, BatchNorm, residual connections)
- **Weight shape transformation** — Correct tensor layout conversion for Conv2D and Dense layers
- **BatchNorm parameter mapping** — Critical for ResNet8 which uses BN after every conv
- **Layer name mapping** — Corresponding layers between Keras and PyTorch naming conventions
- **Numerical validation** — Compare outputs on same inputs (tolerance: 1e-5)
- **Accuracy validation** — Run full CIFAR-10 test set, target >85%

**Should have (quality/reliability):**
- **Per-layer output comparison** — Detect exactly where conversion diverges (use for debugging)
- **Automated test suite** — Catch regressions during conversion iterations
- **State dict serialization** — Save converted PyTorch weights as .pt file for reusability
- **Weight freeze verification** — Ensure eval mode is set correctly
- **Conversion validation report** — Document layer mapping, accuracy metrics, warnings

**Defer (v2+):**
- **ONNX conversion path** — Investigate only if manual conversion fails validation
- **Automated conversion tools** — No reliable tools exist in 2026; manual is standard practice
- **Training from scratch** — Defeats purpose; only needed if conversion fails completely

### Architecture Approach

The conversion follows a **three-component architecture**: (1) Model Definition (PyTorch ResNet8 structure), (2) Weight Converter (extract and transform weights), and (3) Evaluation Script (validate accuracy). This separates concerns cleanly and enables independent testing of each component.

**Major components:**
1. **Model Definition** (`models/resnet8.py`) — PyTorch architecture inheriting from nn.Module, defines layers without weights
2. **Converter Script** (`convert.py`) — Loads Keras .h5, extracts weights as numpy arrays, applies shape transformations, maps to PyTorch state_dict, saves .pt file
3. **Evaluation Script** (`evaluate.py`) — Loads .pt weights, runs inference in eval mode with torch.no_grad(), calculates accuracy on CIFAR-10 test set

**Data flow:** Keras .h5 → Converter extracts weights → Shape transformations → PyTorch state_dict → .pt file → Evaluator loads → Inference → Metrics

**Key pattern:** Validation-driven conversion with multiple checkpoints: layer count match, parameter count match, shape validation, numerical validation (same input → same output), metric validation (accuracy on test set).

### Critical Pitfalls

1. **Channel order mismatch (NHWC vs NCHW)** — Keras uses (H,W,C_in,C_out) for Conv2D weights, PyTorch uses (C_out,C_in,H,W). Failing to transpose causes silent failures with near-random accuracy. **Prevention:** Always transpose Conv2D weights using `.transpose(3,2,0,1)`, Dense weights using `.transpose(1,0)`, and verify first layer output matches.

2. **BatchNorm mode confusion** — Forgetting `model.eval()` before inference causes BatchNorm to use batch statistics instead of learned running statistics, dropping accuracy 5-15%. **Prevention:** Always call `model.eval()` and use `torch.no_grad()` context manager; verify BatchNorm momentum parameter matches (Keras: 0.99, PyTorch: 0.1); set epsilon to 1e-3 to match Keras default.

3. **Residual connection dimension mismatch** — Shortcut connections fail when spatial size changes (stride=2) or channel count increases. Addition operation produces runtime errors or incorrect results. **Prevention:** Implement 1×1 projection convolutions on shortcuts matching stride of residual path; verify input/output shapes match before addition; test stride=2 blocks specifically.

4. **Weight initialization oversight** — Missing or incorrectly initialized layers (shortcuts, projections) cause 10-20% accuracy drops. **Prevention:** Verify ALL layers have loaded weights by printing tensor statistics (min, max, mean, std) for each layer; ensure shortcut/projection layers are included in weight mapping.

5. **Data preprocessing mismatch** — CIFAR-10 preprocessing must match Keras training exactly. Different normalization causes accuracy drops despite correct model. **Prevention:** Verify exact preprocessing from Keras training script (typically divide by 255.0); test with single known image and compare outputs.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Model Definition
**Rationale:** Architecture must be defined before weights can be loaded. This is the foundation for both conversion and evaluation.

**Delivers:** PyTorch ResNet8 class matching Keras architecture exactly (Conv2D layers, BatchNorm, residual connections with shortcuts)

**Addresses:** Architecture reconstruction (table stakes feature)

**Avoids:** Residual connection dimension mismatch pitfall by explicitly implementing shortcuts with 1×1 projections for stride=2 blocks

**Implementation:** Define `models/resnet8.py` with nn.Module class, verify layer count matches Keras model, test instantiation with random weights

### Phase 2: Weight Converter Implementation
**Rationale:** Core conversion logic transforms Keras weights to PyTorch format. This is the most complex phase requiring careful shape transformations.

**Delivers:** `convert.py` script that extracts .h5 weights, applies transformations, saves .pt file

**Uses:** h5py 3.15.1 for weight extraction, numpy for array operations, PyTorch for state_dict creation

**Addresses:** Weight shape transformation, BatchNorm parameter mapping, layer name mapping (table stakes features)

**Avoids:** Channel order mismatch by implementing correct transpose operations; weight initialization oversight by validating ALL layers loaded

**Implementation:** Load Keras model, extract weights layer-by-layer, transform shapes (Conv2D: transpose(3,2,0,1), Dense: transpose(1,0), bias: no change), map to PyTorch names (gamma→weight, beta→bias for BatchNorm), save state_dict

### Phase 3: Validation and Testing
**Rationale:** Conversion correctness must be verified before running full evaluation. Multi-stage validation catches errors early.

**Delivers:** Numerical validation comparing Keras and PyTorch outputs, parameter count verification, shape validation tests

**Addresses:** Numerical validation (table stakes feature)

**Avoids:** Silent failures by comparing outputs on same inputs with small tolerance (1e-5)

**Implementation:** Create test inputs, run through both Keras and PyTorch models, compare outputs with np.testing.assert_allclose, verify parameter counts match

### Phase 4: Evaluation Script
**Rationale:** Final validation against CIFAR-10 test set proves conversion correctness and meets project goal (≥85% accuracy).

**Delivers:** `evaluate.py` script that loads .pt model, runs inference on CIFAR-10 test set, calculates accuracy metrics

**Uses:** torchvision 0.25 for CIFAR-10 dataset, torchmetrics 1.8.2 for accuracy calculation

**Addresses:** Accuracy validation (table stakes feature)

**Avoids:** BatchNorm mode confusion by calling model.eval() and using torch.no_grad(); data preprocessing mismatch by matching Keras normalization exactly

**Implementation:** Load PyTorch model and state_dict, set eval mode, load CIFAR-10 test data with correct preprocessing, run inference, calculate accuracy, compare to 85% threshold

### Phase Ordering Rationale

- **Architecture first:** Model structure is required by both converter and evaluator; no dependencies
- **Conversion second:** Needs architecture to instantiate model and load weights; produces .pt artifact
- **Validation third:** Needs both architecture and conversion working; catches errors before full evaluation
- **Evaluation last:** Consumes .pt file from conversion; validates against project success criteria

This ordering follows the natural dependency chain and enables incremental validation. Each phase can be tested independently before proceeding to the next.

### Research Flags

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Model Definition):** ResNet architecture is well-documented; PyTorch nn.Module patterns are standard
- **Phase 2 (Weight Converter):** Manual conversion workflow is well-established in PyTorch community; multiple reference implementations exist
- **Phase 4 (Evaluation Script):** CIFAR-10 evaluation is canonical example in PyTorch tutorials

**Phases that may benefit from deeper research:**
- **Phase 3 (Validation):** If conversion fails numerical validation, may need research into specific layer incompatibilities or epsilon values
- **Fallback scenario:** If manual conversion proves difficult, research ONNX pipeline (tf2onnx → onnx2torch)

**Overall:** This project has well-documented patterns; research-phase likely not needed during planning unless unexpected issues arise during implementation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PyTorch 2.10.0, torchvision 0.25, h5py 3.15.1 versions verified from official sources; installation paths tested |
| Features | MEDIUM-HIGH | Table stakes features verified across multiple community sources and tutorials; manual conversion is consensus approach |
| Architecture | HIGH | Three-component structure (model, converter, evaluator) is standard PyTorch project pattern; verified in official tutorials and professional templates |
| Pitfalls | MEDIUM | Channel order and BatchNorm issues verified across multiple forum discussions; specific accuracy numbers (5-15% drop) are anecdotal but consistent |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

**During implementation:**
- **Exact ResNet8 layer structure** — Must inspect .h5 file to confirm layer names, counts, and residual block configuration. Research assumed standard ResNet8 but Keras implementation may have variations.
- **Keras training preprocessing** — Need to verify exact normalization used during training (divide by 255.0 vs dataset mean/std subtraction). Critical for evaluation accuracy.
- **Epsilon values for BatchNorm** — Confirm Keras model used 1e-3 (default) or custom value. Small differences accumulate in deep networks.

**During validation:**
- **Acceptable tolerance thresholds** — Research suggests 1e-5 for numerical validation, but may need adjustment based on actual precision drift observed.
- **Baseline Keras accuracy** — Don't have confirmed accuracy of source Keras model on CIFAR-10. Need to establish baseline before validating PyTorch conversion.

**If manual conversion fails:**
- **ONNX operator support** — onnx2torch "covers only a limited number of PyTorch/ONNX models and operations." May need custom operator registration for edge cases.
- **Custom layer handling** — If ResNet8 uses non-standard layers, may need additional research into conversion approaches.

These gaps are expected for pre-implementation research and will be resolved during Phase 1 (inspect .h5 file) and Phase 3 (validation testing).

## Sources

### Primary (HIGH confidence)
- PyTorch Installation Guide — Version 2.10.0 verification
- PyTorch torchvision Documentation — CIFAR-10 dataset loader, version 0.25
- h5py PyPI — Version 3.15.1 (Oct 2025)
- PyTorch Saving and Loading Models Tutorial — State_dict patterns
- PyTorch Model Definition Tutorial — nn.Module architecture patterns
- MLCommons TinyMLPerf Keras Model — ResNet8 reference implementation

### Secondary (MEDIUM confidence)
- PyTorch Forums: Keras to PyTorch Conversion — Manual conversion workflows, community consensus
- PyTorch Forums: Transferring Weights from Keras — Shape transformation patterns
- Medium: Load Keras Weight to PyTorch — Layer-by-layer mapping approach
- Medium: Challenge of Converting TensorFlow to PyTorch — ONNX vs manual comparison
- GitHub: onnx2torch — Version 1.5.15, active maintenance status
- GitHub: tf2onnx — Version 1.16.1, Keras to ONNX conversion
- Multiple sources on BatchNorm pitfalls — Eval mode, epsilon differences, momentum parameters

### Tertiary (LOW confidence — flagged for validation)
- Specific accuracy degradation percentages (98%→60%, 5-15% drop) from forum discussions
- ONNX as "fallback only" recommendation — some sources recommend ONNX as primary path
- Padding calculation formulas — "same" vs integer padding equivalence needs verification during implementation

---
*Research completed: 2026-01-27*
*Ready for roadmap: yes*
