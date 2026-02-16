---
status: testing
phase: 15-intermediate-activation-capture
source: [15-01-SUMMARY.md]
started: 2026-02-16T00:00:00Z
updated: 2026-02-16T00:00:00Z
---

## Current Test

number: 1
name: Weight visualization regression check
expected: |
  Load a .pt model in the notebook, select a layer and tensor. The weight histogram (blue) and statistics panel should display exactly as before — no regressions from the new activation features.
awaiting: user response

## Tests

### 1. Weight visualization regression check
expected: Load a .pt model, select a layer and tensor. Weight histogram (blue) and stats panel display correctly — same as before activation features were added.
result: [pending]

### 2. Input source selection UI appears for PyTorch model
expected: With a .pt model selected, an "Intermediate Activations" section appears below model loading. It shows a radio button (CIFAR-10 sample / Random input), a sample index number input, and a "Run Inference" button.
result: [pending]

### 3. Activation capture with random input
expected: Select "Random input", click "Run Inference". A spinner shows briefly, then a green success callout appears saying "Activations captured: N tensors | Input: random (1, 32, 32, 3)" where N is some number of captured tensors.
result: [pending]

### 4. View toggle and activation histogram
expected: After capturing activations, a "View" radio button appears with "Weights" and "Activations" options. Switching to "Activations" changes the histogram color from blue to orange, title changes to "Activation Distribution", and stats panel header changes to "Activation Statistics".
result: [pending]

### 5. Toggle back to weights
expected: Switching the view toggle back to "Weights" restores the blue histogram, "Weight Distribution" title, and "Weight Statistics" header. Data matches the original weight view.
result: [pending]

### 6. ONNX model hides activation features
expected: Select an ONNX model (.onnx file). The activation input source UI is replaced with a message like "Activation capture requires a PyTorch model (.pt)". No run button, no view toggle.
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0

## Gaps

[none yet]
