## MODIFIED Requirements

### Requirement: Unified evaluation pipeline contract
The system SHALL provide a shared evaluation pipeline used by both ONNX and
PyTorch evaluation entry points for CIFAR-10 test loading, prediction
aggregation, and metric computation.

#### Scenario: ONNX and PyTorch use the same evaluation flow
- **WHEN** a user runs ONNX and PyTorch evaluation commands against the same
  CIFAR-10 test set
- **THEN** both commands MUST execute the same shared data-loading and
  metric-computation path, with backend-specific logic isolated to inference
  adapters

#### Scenario: PyTorch evaluation supports PTQ bit-width overrides
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq`
- **THEN** the PyTorch adapter MUST apply PTQ simulation using the provided
  bit-width settings during inference
- **AND** existing evaluation reporting and metric computation MUST remain
  unchanged

#### Scenario: PyTorch evaluation supports calibrated PTQ mode
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and/or `--aq`
  and `--calib`
- **THEN** the PyTorch adapter MUST derive quantization parameters from
  calibration data loaded from `<data-dir>/test_batch`
- **AND** those calibrated parameters MUST be applied during inference-time PTQ
  simulation
- **AND** evaluation reporting and metric computation MUST remain unchanged

#### Scenario: PyTorch evaluation selects activation quantization scheme
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--aq` and
  `--aq-scheme asymmetric`
- **THEN** the PyTorch adapter MUST apply asymmetric fake quantization for
  activation tensors
- **AND** weight PTQ behavior MUST remain symmetric
- **AND** existing evaluation reporting and metric computation MUST remain
  unchanged

#### Scenario: Calibrated asymmetric activation quantization
- **WHEN** a user enables `--calib` with `--aq-scheme asymmetric`
- **THEN** activation quantization parameters MUST be derived from calibration
  activation ranges
- **AND** those fixed parameters MUST be used during inference PTQ simulation

#### Scenario: PyTorch evaluation supports empirical DFQ bias correction
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and
  `--dfq-bias-corr`
- **THEN** the PyTorch adapter MUST apply empirical per-layer bias correction
  after weight PTQ using calibration images from `<data-dir>/test_batch`
- **AND** quantization summary output MUST indicate corrected weight layers

#### Scenario: DFQ bias correction requires weight PTQ
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--dfq-bias-corr`
  without `--wq`
- **THEN** the command MUST fail fast with a clear argument error

#### Scenario: Default PTQ behavior is preserved without DFQ option
- **WHEN** a user runs `scripts/evaluate_pytorch.py` with `--wq` and without
  `--dfq-bias-corr`
- **THEN** weight PTQ behavior MUST remain unchanged from prior per-tensor/per-
  channel simulation semantics

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
