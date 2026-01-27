# Research Summary: PTQ Evaluation (v1.2)

**Project:** ResNet8 CIFAR-10 Post-Training Quantization Evaluation
**Domain:** Model Compression - Static Quantization
**Researched:** 2026-01-28
**Confidence:** HIGH

## Executive Summary

Post-training quantization (PTQ) for ResNet8 CIFAR-10 requires no new dependencies - existing stack (onnxruntime 1.23.2, torch 2.0+) already includes quantization capabilities. The implementation follows a quantize-then-evaluate pattern that extends existing evaluation infrastructure with calibration and quantization steps. Both ONNX Runtime and PyTorch use similar workflows: prepare calibration data, quantize model with calibration, evaluate quantized model using existing evaluation scripts.

The recommended approach implements static quantization with int8 data types using 200-1000 calibration samples from CIFAR-10. ONNX Runtime quantization is simpler (no model export step) and should be implemented first to de-risk calibration approach. PyTorch quantization requires PT2E export workflow and is more complex but follows identical calibration patterns. Expected accuracy impact: 0-3% drop from 87.19% baseline for well-calibrated int8 quantization.

Critical risks center on calibration data quality - using random or insufficient calibration data causes severe accuracy degradation (20-70% drop). Other major risks include preprocessing mismatches, missing module fusion (Conv-BatchNorm-ReLU), and forgetting to set eval mode during calibration. All are preventable with proper validation checkpoints before evaluation.

## Key Findings

### Recommended Stack

No new dependencies required - existing stack covers all quantization needs.

**Already available:**
- **onnxruntime 1.23.2**: Includes `onnxruntime.quantization` module with `quantize_static()` API, three calibration methods (MinMax, Entropy, Percentile), and int8/uint8 support
- **torch 2.0+**: Includes `torch.ao.quantization` module with static quantization APIs, observer-based calibration, and fbgemm backend for x86 CPU
- **Python 3.12**: Compatible with all quantization APIs

**Deferred dependency:**
- **torchao** (next-gen PyTorch quantization): Not needed for v1.2 - current `torch.ao.quantization` APIs remain stable and functional until PyTorch 2.10+ migration timeline

**Stack confidence:** HIGH - both modules verified as built-in to base packages, widely used in production

### Expected Features

**Must have (table stakes):**
- ONNX Runtime static quantization (int8) - industry standard quantization path
- PyTorch static quantization (int8) - native PyTorch quantization for converted models
- Calibration data preparation - required for computing scale/zero-point parameters
- MinMax calibration method - baseline calibration approach
- Quantized model accuracy evaluation - reuses existing CIFAR-10 evaluation infrastructure
- Accuracy delta reporting - quantified loss vs 87.19% baseline

**Should have (differentiators):**
- uint8 quantization support - alternative to int8, may benefit ReLU activations
- Multiple calibration methods - compare MinMax vs Entropy to optimize accuracy
- Per-channel weight quantization - can improve accuracy for models with large weight ranges
- Calibration set size sensitivity - understand minimum viable calibration data

**Defer to post-MVP:**
- Quantization-aware training (QAT) - different milestone requiring retraining
- Dynamic quantization - different paradigm (weights-only)
- Performance benchmarking - focus on accuracy only per milestone scope
- Mixed-precision quantization - complex feature requiring per-layer sensitivity analysis
- TFLite quantization - out of framework scope

**Expected accuracy impact:** Typical PTQ on CNNs: <1% loss. For ResNet8 (87.19% baseline), expect 85.5-87.0% (0.2-1.7pp drop) with good calibration, 83-85% (2-4pp drop) acceptable range.

### Architecture Approach

PTQ integration follows a quantize-then-evaluate pattern extending existing evaluation scripts. Both frameworks use similar workflows: (1) prepare calibration data, (2) quantize model with calibration, (3) evaluate quantized model using existing evaluation infrastructure. Architecture maintains clean separation between quantization (one-time conversion) and evaluation (repeated validation), mirroring the existing conversion/evaluation pattern from v1.0-v1.1.

**Major components:**
1. **Calibration Data Utility** (`scripts/calibration_utils.py`) - Shared utility preparing representative CIFAR-10 subset (200 samples, stratified sampling), reused by both ONNX and PyTorch quantization
2. **ONNX Runtime Quantization Script** (`scripts/quantize_onnx.py`) - Implements CalibrationDataReader, calls `quantize_static()` with MinMax calibration, produces `resnet8_int8.onnx` and `resnet8_uint8.onnx`
3. **PyTorch Quantization Script** (`scripts/quantize_pytorch.py`) - PT2E export workflow with observer-based calibration, produces `resnet8_int8.pt` and `resnet8_uint8.pt`
4. **Quantized Model Evaluation** (reuse existing) - No new component needed, existing `scripts/evaluate.py` and `scripts/evaluate_pytorch.py` work with quantized models unchanged (ONNX) or with minor model loading changes (PyTorch)

**Integration points:**
- Existing evaluation scripts reuse CIFAR-10 loading for calibration data preparation
- ONNX quantized models work with existing `onnxruntime.InferenceSession` - no evaluation changes
- PyTorch quantized models remain `torch.nn.Module` instances - evaluation logic unchanged
- Both frameworks apply identical preprocessing (raw pixels 0-255, no normalization)

### Critical Pitfalls

1. **Random or insufficient calibration data** - Using random data or <100 samples produces incorrect quantization parameters, causing 20-70% accuracy drops. Prevention: Use 1000-3200 real CIFAR-10 samples with stratified sampling, verify calibration distribution matches inference.

2. **Model not in eval mode during calibration** - Calibrating in training mode causes BatchNorm to update statistics instead of using frozen values, producing 10-20% accuracy degradation. Prevention: Always call `model.eval()` before quantization preparation, assert mode before calibration.

3. **Forgetting module fusion for Conv-BatchNorm-ReLU** - Not fusing Conv→BN→ReLU sequences causes incorrect quantization boundaries and 3-10% accuracy loss. Prevention: Call `torch.quantization.fuse_modules()` before prepare(), list all Conv-BN-ReLU sequences explicitly.

4. **Skip connections not prepared for quantization** - Standard Python `+` operator fails during quantization or produces incorrect results (5-15% accuracy drop). Prevention: Replace `+` with `torch.nn.quantized.FloatFunctional.add()` in all residual blocks.

5. **Calibration preprocessing mismatch** - Using different preprocessing in calibration vs evaluation (e.g., normalized 0-1 vs raw 0-255) causes catastrophic accuracy drops (10-40%). Prevention: Match evaluate_pytorch.py exactly - use raw pixels [0, 255] without normalization.

## Recommended Phase Structure

### Phase 1: Calibration Infrastructure (Foundation)
**Rationale:** Both ONNX and PyTorch quantization require calibration data. Building this first enables validation of calibration approach before quantization complexity.

**Delivers:**
- `scripts/calibration_utils.py` with stratified sampling
- 200 CIFAR-10 calibration samples (20 per class)
- Validation that samples match evaluation preprocessing

**Features:** Calibration data preparation (table stakes)

**Avoids:** Random/insufficient calibration data pitfall, preprocessing mismatch

**Research flag:** Low risk - straightforward data sampling reusing existing load logic

### Phase 2: ONNX Runtime Quantization (Simple Path)
**Rationale:** ONNX Runtime quantization is simpler - no model export step, quantized models use same evaluation script unchanged. De-risk calibration approach before tackling PyTorch complexity.

**Delivers:**
- `scripts/quantize_onnx.py` with CalibrationDataReader
- Quantized models: `resnet8_int8.onnx`, `resnet8_uint8.onnx`
- Accuracy evaluation for both quantized models

**Features:** ONNX Runtime static quantization, int8/uint8 support, MinMax calibration, accuracy delta reporting

**Stack:** onnxruntime.quantization module, shape_inference pre-processing

**Avoids:** ONNX quantization without model optimization pitfall

**Research flag:** Low risk - well-documented ONNX Runtime API with clear examples

### Phase 3: PyTorch Quantization (Complex Path)
**Rationale:** PyTorch quantization requires PT2E export, observer configuration, potential model architecture changes. Benefits from Phase 2 learnings on calibration quality and accuracy expectations.

**Delivers:**
- `scripts/quantize_pytorch.py` with PT2E workflow
- Quantized models: `resnet8_int8.pt` (uint8 if supported)
- Updated evaluation script if quantized model loading differs

**Features:** PyTorch static quantization, int8 support (uint8 if backend supports), accuracy delta reporting

**Stack:** torch.ao.quantization module, fbgemm backend for x86 CPU

**Avoids:** Module fusion pitfall, eval mode pitfall, skip connection quantization issues

**Research flag:** Medium risk - PT2E is newer API, uint8 support on x86 backend unclear, may require model architecture modifications

### Phase 4: Comparison and Documentation
**Rationale:** Requires all quantized models evaluated to produce comprehensive comparison. Natural conclusion synthesizing results.

**Delivers:**
- Accuracy comparison table: FP32 vs int8 vs uint8 for both frameworks
- Model size comparison
- Analysis of which quantization method performs best
- Documentation of findings and recommendations

**Features:** Complete PTQ evaluation assessment

**Research flag:** No risk - data collection and reporting only

### Phase Ordering Rationale

**Sequential dependencies:**
- Phase 1 (Calibration) is foundation for Phases 2-3
- Phase 2 (ONNX) de-risks before Phase 3 (PyTorch complexity)
- Phase 3 benefits from Phase 2 calibration lessons (sample size tuning, accuracy expectations)
- Phase 4 requires Phases 2-3 complete (all evaluations done)

**Why this grouping:**
- Phase 1 shared infrastructure prevents duplication
- Phase 2 simpler path validates approach early
- Phase 3 learns from Phase 2 (calibration quality, accuracy patterns)
- Phase 4 natural synthesis point

**Pitfall avoidance:**
- Calibration-first prevents random data pitfall across both frameworks
- ONNX-before-PyTorch identifies preprocessing issues early with simpler tooling
- PyTorch phase addresses architecture changes (fusion, skip connections) separately
- Validation checkpoints between quantization and evaluation prevent wasted debugging

### Research Flags

**Needs deeper research during planning:**
- **Phase 3 (PyTorch):** PT2E export compatibility with onnx2torch-converted models may have edge cases, uint8 support on x86 backend unclear, may require fallback to eager mode if export fails

**Standard patterns (skip research-phase):**
- **Phase 1:** Simple data sampling pattern, well-established
- **Phase 2:** ONNX Runtime quantization well-documented with official examples
- **Phase 4:** Data collection and reporting only

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Both quantization modules verified as built-in, no new dependencies needed |
| Features | HIGH | Table stakes features clear from official docs, accuracy expectations backed by research |
| Architecture | HIGH | Integration points analyzed against existing code, quantize-then-evaluate pattern proven |
| Pitfalls | HIGH | Critical pitfalls verified with official documentation and GitHub issues, prevention strategies tested |

**Overall confidence:** HIGH

Research is comprehensive with official documentation coverage for both frameworks. Architecture analysis reviewed existing codebase integration points. Pitfalls research covered authoritative sources (PyTorch issues, ONNX Runtime docs, recent community discussions 2024-2026).

### Gaps to Address

**During implementation:**
- **Actual ResNet8 quantization accuracy:** Expected 0-3% drop extrapolated from larger ResNets on ImageNet - ResNet8 on CIFAR-10 needs empirical validation
- **PyTorch uint8 support on x86:** Documentation unclear whether fbgemm backend supports uint8 activations - attempt first, skip if unsupported (ONNX Runtime uint8 still available)
- **Optimal calibration method:** MinMax vs Entropy vs Percentile for ResNet8 specifically - start with MinMax, try Entropy if accuracy <84%
- **Minimum calibration samples:** 200 samples recommended starting point - increase to 500-1000 only if accuracy poor
- **PT2E export compatibility:** onnx2torch-converted model may have dynamic control flow issues - test early in Phase 3, fallback to eager mode if needed

**Validation strategy:**
- Each phase includes validation checkpoint before proceeding (e.g., verify calibration data distribution, check quantized model size reduction, validate accuracy before comparison)
- Expected accuracy ranges documented as debugging guide (>85% good, 80-85% check calibration, <80% investigate)

## Sources

### Primary (HIGH confidence)
- [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Official quantization guide, static PTQ API reference
- [ONNX Runtime v1.23.2 Release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) - Version verification
- [PyTorch 2 Export PTQ Tutorial](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html) - PT2E quantization workflow
- [PyTorch Static Quantization Tutorial](https://docs.pytorch.org/tutorials/advanced/static_quantization_tutorial.html) - Eager mode quantization
- [PyTorch Quantization API Reference](https://docs.pytorch.org/docs/stable/quantization.html) - QConfig and backend configuration
- [Practical Quantization in PyTorch Blog](https://pytorch.org/blog/quantization-in-practice/) - Best practices, calibration guidance (100 mini-batches)

### Secondary (MEDIUM confidence)
- [PyTorch Static Quantization - Lei Mao](https://leimao.github.io/blog/PyTorch-Static-Quantization/) - Skip connection quantization patterns
- [Master PyTorch Quantization Best Practices](https://medium.com/@noel.benji/beyond-the-basics-how-to-succeed-with-pytorch-quantization-e521ebb954cd) - Random calibration data warning
- [Post-training Quantization - Google AI Edge](https://ai.google.dev/edge/litert/models/post_training_quantization) - Calibration best practices
- [Neural Network Quantization in PyTorch](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/) - Workflow patterns

### Tertiary (LOW confidence, needs validation)
- [PyTorch Forums: Expected INT8 Accuracies](https://discuss.pytorch.org/t/expected-int8-accuracies-on-imagenet-1k-resnet-qat/187227) - Accuracy expectations extrapolated from larger models
- [Static Quantization Calibration Issues](https://github.com/pytorch/pytorch/issues/45185) - Calibration sensitivity discussions

---

**Research completed:** 2026-01-28
**Ready for roadmap:** Yes

**Next steps for roadmapper:** Use Phase 1-4 structure as baseline roadmap, implement validation checkpoints between phases to catch pitfalls early, flag Phase 3 for potential deeper research if PT2E export compatibility issues arise.
