## MODIFIED Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and PyTorch evaluation entry points for CIFAR-10 test loading, prediction aggregation, and metric computation.

#### Scenario: PyTorch evaluation supports PTQ bit-width overrides
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq`
- **THEN** the PyTorch adapter MUST apply PTQ simulation using the provided bit-width settings during inference
- **AND** existing evaluation reporting and metric computation MUST remain unchanged

### Requirement: Explicit preprocessing consistency
The system MUST centralize and document preprocessing rules used by evaluation so ONNX and PyTorch backends apply identical input representation semantics.

#### Scenario: Quantized PyTorch evaluation preserves shared preprocessing
- **WHEN** PTQ simulation is enabled for PyTorch evaluation
- **THEN** CIFAR-10 data loading and preprocessing MUST continue to use the shared evaluation pipeline without backend-specific duplication
