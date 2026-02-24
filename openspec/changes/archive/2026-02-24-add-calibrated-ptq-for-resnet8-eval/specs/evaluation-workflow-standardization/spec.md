## MODIFIED Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and PyTorch evaluation entry points for CIFAR-10 test loading, prediction aggregation, and metric computation.

#### Scenario: PyTorch evaluation supports calibrated PTQ mode
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq` and `--calib`
- **THEN** the PyTorch adapter MUST derive quantization parameters from calibration data loaded from `<data-dir>/test_batch`
- **AND** those calibrated parameters MUST be applied during inference-time PTQ simulation
- **AND** evaluation reporting and metric computation MUST remain unchanged

### Requirement: Explicit preprocessing consistency
The system MUST centralize and document preprocessing rules used by evaluation so ONNX and PyTorch backends apply identical input representation semantics.

#### Scenario: Calibrated PTQ uses shared CIFAR preprocessing semantics
- **WHEN** calibrated PTQ mode is enabled
- **THEN** calibration input parsing and tensor layout handling MUST match evaluation pipeline expectations for CIFAR-10 test data
- **AND** non-calibrated evaluation behavior MUST remain backward compatible
