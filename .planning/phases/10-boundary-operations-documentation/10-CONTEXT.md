# Phase 10: Boundary Operations Documentation — Context

> Decisions from discussion session. Guides research and planning.

## Formula Presentation

- **Format:** Markdown (not LaTeX-heavy)
- **Explanation style:** Describe how each formula is formed, but show the formula directly without step-by-step derivation
- **Coverage:** Both symmetric (zero_point = 0) and asymmetric quantization cases
- **Round-trip relationship:** Include a section proving `Dequant(Quant(x)) ≈ x` with error bounds
- **Variable naming:** Use ONNX spec names exactly (`x`, `y_scale`, `y_zero_point`)

## Numerical Examples

- No specific constraints — use judgment for clarity

## Hardware Pseudocode

- **Skip this section entirely** — integer matmul documentation is sufficient
- Do not include C-style or Verilog snippets for boundary operations

## Document Structure

- **Organization:** Operation-by-operation (QuantizeLinear first, then DequantizeLinear)
- Each operation as its own section with formula, cases, and examples

## Deferred Ideas

None.
