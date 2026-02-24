## MODIFIED Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and PyTorch evaluation entry points for CIFAR-10 test loading, prediction aggregation, and metric computation.

#### Scenario: PyTorch evaluation selects activation quantization scheme
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--aq` and `--aq-scheme asymmetric`
- **THEN** the PyTorch adapter MUST apply asymmetric fake quantization for activation tensors
- **AND** weight PTQ behavior MUST remain symmetric
- **AND** existing evaluation reporting and metric computation MUST remain unchanged

#### Scenario: Calibrated asymmetric activation quantization
- **WHEN** a user enables `--calib` with `--aq-scheme asymmetric`
- **THEN** activation quantization parameters MUST be derived from calibration activation ranges
- **AND** those fixed parameters MUST be used during inference PTQ simulation

### Requirement: Explicit preprocessing consistency
The system MUST centralize and document preprocessing rules used by evaluation so ONNX and PyTorch backends apply identical input representation semantics.

#### Scenario: Asymmetric activation mode preserves shared preprocessing
- **WHEN** asymmetric activation quantization is enabled
- **THEN** CIFAR-10 loading and preprocessing MUST continue to use the shared evaluation pipeline
