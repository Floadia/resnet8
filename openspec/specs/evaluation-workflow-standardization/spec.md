## Purpose

Standardize the ResNet8 evaluation workflow so ONNX and PyTorch paths share
preprocessing, metrics, and deterministic reporting behavior.

## Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and
PyTorch evaluation entry points for CIFAR-10 test loading, prediction
aggregation, and metric computation.

#### Scenario: ONNX and PyTorch use the same evaluation flow
- **WHEN** a user runs ONNX and PyTorch evaluation commands against the same CIFAR-10 test set
- **THEN** both commands MUST execute the same shared data-loading and metric-computation path, with backend-specific logic isolated to inference adapters

#### Scenario: PyTorch evaluation supports PTQ bit-width overrides
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq`
- **THEN** the PyTorch adapter MUST apply PTQ simulation using the provided bit-width settings during inference
- **AND** existing evaluation reporting and metric computation MUST remain unchanged

#### Scenario: PyTorch evaluation supports calibrated PTQ mode
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq` and `--calib`
- **THEN** the PyTorch adapter MUST derive quantization parameters from calibration data loaded from `<data-dir>/test_batch`
- **AND** those calibrated parameters MUST be applied during inference-time PTQ simulation
- **AND** evaluation reporting and metric computation MUST remain unchanged

#### Scenario: PyTorch evaluation selects activation quantization scheme
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--aq` and `--aq-scheme asymmetric`
- **THEN** the PyTorch adapter MUST apply asymmetric fake quantization for activation tensors
- **AND** weight PTQ behavior MUST remain symmetric
- **AND** existing evaluation reporting and metric computation MUST remain unchanged

#### Scenario: Calibrated asymmetric activation quantization
- **WHEN** a user enables `--calib` with `--aq-scheme asymmetric`
- **THEN** activation quantization parameters MUST be derived from calibration activation ranges
- **AND** those fixed parameters MUST be used during inference PTQ simulation

### Requirement: Deterministic evaluation report schema
The system SHALL emit a deterministic report schema containing overall accuracy,
per-class accuracy, sample counts, and execution metadata (backend and model
path).

#### Scenario: Evaluation output keys are stable
- **WHEN** evaluation completes successfully
- **THEN** the produced report MUST include stable field names for overall metrics, per-class metrics, and metadata so downstream tooling can parse it without backend-specific branching

### Requirement: Explicit preprocessing consistency
The system MUST centralize and document preprocessing rules used by evaluation
so ONNX and PyTorch backends apply identical input representation semantics.

#### Scenario: Shared preprocessing source of truth
- **WHEN** preprocessing logic is updated
- **THEN** both evaluation backends MUST automatically inherit the change from the shared preprocessing implementation and must not duplicate preprocessing logic in backend scripts

#### Scenario: Quantized PyTorch evaluation preserves shared preprocessing
- **WHEN** PTQ simulation is enabled for PyTorch evaluation
- **THEN** CIFAR-10 data loading and preprocessing MUST continue to use the shared evaluation pipeline without backend-specific duplication

#### Scenario: Calibrated PTQ uses shared CIFAR preprocessing semantics
- **WHEN** calibrated PTQ mode is enabled
- **THEN** calibration input parsing and tensor layout handling MUST match evaluation pipeline expectations for CIFAR-10 test data
- **AND** non-calibrated evaluation behavior MUST remain backward compatible

#### Scenario: Asymmetric activation mode preserves shared preprocessing
- **WHEN** asymmetric activation quantization is enabled
- **THEN** CIFAR-10 loading and preprocessing MUST continue to use the shared evaluation pipeline
