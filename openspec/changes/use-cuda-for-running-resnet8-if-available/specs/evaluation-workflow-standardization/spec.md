## MODIFIED Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and
PyTorch evaluation entry points for CIFAR-10 test loading, prediction
aggregation, and metric computation.

#### Scenario: PyTorch evaluation auto-selects CUDA when available
- **WHEN** a user runs `scripts/evaluate_pytorch.py` without an explicit device
  override
- **THEN** the PyTorch evaluation path MUST select `cuda` when
  `torch.cuda.is_available()` is true
- **AND** it MUST fall back to `cpu` when CUDA is unavailable
- **AND** model and inference tensors MUST be placed on the selected device

#### Scenario: PyTorch evaluation supports explicit device override
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--device cpu`
- **THEN** evaluation MUST run on CPU even when CUDA is available
- **AND** existing evaluation reporting and metric computation MUST remain
  unchanged

#### Scenario: Explicit CUDA request on non-CUDA host
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--device cuda` on a
  host where CUDA is unavailable
- **THEN** evaluation MUST fail fast with a clear error indicating CUDA is not
  available
