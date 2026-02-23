## Purpose

Standardize ResNet8 weight-visualization data extraction so UI rendering is
reproducible and testable across ONNX and PyTorch models.

## Requirements

### Requirement: Reusable tensor extraction interface
The system SHALL expose reusable tensor-extraction utilities for ONNX and
PyTorch models that return a normalized tensor metadata structure for
visualization.

#### Scenario: Visualizer reads normalized tensor metadata
- **WHEN** a model is loaded for weight visualization
- **THEN** tensor metadata MUST be provided through a shared interface that includes layer identity, tensor type, shape, value arrays, and quantization flags

### Requirement: Quantization metadata fidelity
The system MUST preserve quantization-relevant metadata (integer values, scale,
zero-point, and dequantized values when available) in the visualization data
model.

#### Scenario: Quantized layer inspection
- **WHEN** a user selects a quantized weight tensor
- **THEN** the visualization backend MUST provide both integer-domain and dequantized-domain information required for accurate histogram and statistics rendering

### Requirement: UI-logic separation for visualizer
The system SHALL separate visualization business logic (model parsing and tensor
preparation) from marimo UI wiring so core behavior can be validated without
notebook runtime.

#### Scenario: Core visualizer logic is testable without marimo
- **WHEN** automated tests run for weight-visualization behavior
- **THEN** tests MUST be able to validate tensor parsing and metadata generation without importing or executing marimo UI cells
