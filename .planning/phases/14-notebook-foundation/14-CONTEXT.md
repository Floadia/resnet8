# Phase 14: Notebook Foundation — Context

## Overview
Interactive Marimo notebook for loading quantized model variants and comparing layer distributions across precision levels (float, int8, uint8).

---

## Layout Decisions

| Decision | Choice |
|----------|--------|
| Page structure | Single scrolling page (no tabs) |
| Density | Minimal with whitespace |
| Model metadata shown | Name, format, layer count |
| Utility modules | OK to import from separate files |
| Comparison layout | Horizontal side-by-side |
| Distribution display | Both histograms AND summary stats |
| Variants compared | float, int8, uint8 (precision comparison) |
| Plot organization | Separate plots for input, weight, output |

---

## Model Loading Decisions

| Decision | Choice |
|----------|--------|
| Model selection | File picker dialog |
| Loading behavior | Load all 3 variants together as a set |
| Loading indicator | Spinner (use Marimo built-in) |
| Error display | Inline message (simple, no tracebacks) |

---

## Layer Selection Decisions

| Decision | Choice |
|----------|--------|
| Selection widget | Single-select (one layer at a time) |
| Layer naming | Full path (e.g., `layer1.conv1.weight`) |
| Update trigger | Immediate on selection (no button) |

---

## Initial State Decisions

| Decision | Choice |
|----------|--------|
| Before model load | Instructions text ("Select a model folder to begin") |
| After load, no layer selected | Dropdown shows "Select a layer..." with empty plots |
| Model summary | Show after load (layer count, available layers) |
| On new model load | Reset to "Select a layer..." (no persistence) |

---

## Deferred Ideas
*(Captured during discussion but out of scope for this phase)*

None identified.

---

## Research Hints
- Investigate Marimo's file picker API and built-in spinner/loading components
- Check how to extract layer names with full paths from ONNX and PyTorch models
- Research Marimo's reactive update patterns for immediate selection response
