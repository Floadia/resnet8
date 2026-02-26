## Purpose

Standardize the ResNet8 evaluation workflow so ONNX and PyTorch paths share
preprocessing, metrics, and deterministic reporting behavior.
## Requirements
### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and
PyTorch evaluation entry points for CIFAR-10 test loading, prediction
aggregation, and metric computation.

#### Scenario: PyTorch PTQ evaluation exports eval-quantized ONNX graph
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq`
  and `--export-onnx <path>`
- **THEN** the command MUST export an ONNX model representing the configured
  eval quantization behavior
- **AND** the default evaluation report generation MUST remain unchanged

#### Scenario: Exported ONNX score is checked against PyTorch score
- **WHEN** a user enables `--verify-exported-onnx` together with
  `--export-onnx <path>`
- **THEN** the workflow MUST evaluate the exported ONNX file with the shared
  evaluation pipeline
- **AND** it MUST report absolute accuracy delta versus the PyTorch evaluation
  result
- **AND** it MUST fail with a clear error if delta exceeds the configured
  tolerance

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

