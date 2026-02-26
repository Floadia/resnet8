## MODIFIED Requirements

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
