# Phase 10: Boundary Operations Documentation - Research

**Researched:** 2026-02-02
**Domain:** ONNX quantization operations specification and technical documentation
**Confidence:** HIGH

## Summary

This research investigates how to document ONNX QuantizeLinear and DequantizeLinear operations for hardware implementation. The ONNX specification provides authoritative formulas using specific variable naming conventions (x, y_scale, y_zero_point). GitHub natively supports LaTeX math rendering via MathJax since May 2022, enabling clean markdown-based documentation without external rendering.

The standard approach is operation-by-operation documentation with exact formulas, parameter definitions, saturation ranges, and worked numerical examples. Based on user decisions in CONTEXT.md, hardware pseudocode sections are skipped entirely, and formulas are presented in markdown with direct formula display (not step-by-step derivations).

**Primary recommendation:** Use ONNX operator specification as the authoritative source, GitHub markdown math syntax for formulas, and operation-specific sections with symmetric/asymmetric quantization cases clearly distinguished.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ONNX | 1.21.0 | Operator specification | Official standard for QuantizeLinear/DequantizeLinear definitions |
| GitHub Markdown | Native (May 2022+) | Math rendering | Built-in MathJax support, no external dependencies |
| ONNX Runtime | Latest | Reference implementation | Validates formula correctness via actual behavior |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| N/A | N/A | N/A | Documentation phase uses standard markdown only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| GitHub Markdown | Jupyter notebooks with LaTeX | More complex rendering but harder to version/review |
| Inline math $...$ | Display math $$...$$ | Inline for small expressions, display for main formulas |
| Direct LaTeX | Rendered images | Images break accessibility and are harder to maintain |

**Installation:**
```bash
# No installation required - uses GitHub native markdown rendering
# For local preview, any markdown viewer with MathJax support
```

## Architecture Patterns

### Recommended Documentation Structure
```
docs/
├── quantization/
│   ├── 01-boundary-operations.md    # QuantizeLinear + DequantizeLinear
│   ├── 02-core-operations.md        # QLinearConv, QLinearMatMul
│   └── 03-architecture.md           # Data flow and integration
```

### Pattern 1: Operation-First Organization
**What:** Each operation gets its own major section with complete specification
**When to use:** Documenting independent operations that can be understood in isolation
**Example:**
```markdown
# QuantizeLinear Operation

## Overview
Brief description of what the operation does

## Formula
$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

## Parameters
### Inputs
- **x**: Input tensor to quantize (float32/float16/bfloat16)
- **y_scale**: Scale factor (scalar or tensor)
- **y_zero_point**: Zero point offset (optional, defaults to 0)

### Output
- **y**: Quantized tensor (int8/uint8/int16/uint16)

## Cases

### Symmetric Quantization (y_zero_point = 0)
...

### Asymmetric Quantization
...

## Numerical Example
...
```
**Source:** ONNX operator specification pattern (https://onnx.ai/onnx/operators/)

### Pattern 2: Formula-First with Context
**What:** Lead with the exact formula, then explain each component
**When to use:** When readers need to quickly reference the formula and understand parameters
**Example:**
```markdown
## Formula

The quantization operation follows:

$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

Where:
- Division uses **round-to-nearest-even** (banker's rounding)
- Saturation clips to data type range: [0, 255] for uint8, [-128, 127] for int8

## How the Formula is Formed

The scale factor `y_scale` determines the quantization step size...
```
**Source:** ONNX Runtime quantization documentation pattern

### Pattern 3: Symmetric/Asymmetric Case Split
**What:** Separate sections for zero_point=0 (symmetric) vs non-zero (asymmetric)
**When to use:** When the two cases have different semantics or hardware implications
**Example:**
```markdown
## Symmetric Quantization (zero_point = 0)

Formula simplifies to:
$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right)\right)$$

**Characteristics:**
- Zero maps exactly to 0 in quantized space
- Symmetric range around zero: [-127, 127] for int8
- Simpler hardware (no zero-point addition)

## Asymmetric Quantization (zero_point ≠ 0)

Full formula:
$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

**Characteristics:**
- Zero can map to any quantized value
- Full range utilization: [-128, 127] for int8 or [0, 255] for uint8
- Better accuracy for asymmetric distributions
```
**Source:** ONNX Runtime quantization concepts

### Pattern 4: Round-Trip Error Analysis
**What:** Document the error bounds when quantizing then dequantizing
**When to use:** When users need to understand quantization accuracy loss
**Example:**
```markdown
## Round-Trip Relationship

For any input `x`:
$$\text{Dequant}(\text{Quant}(x)) \approx x$$

### Error Bound

The maximum round-trip error is bounded by:
$$|x - \text{Dequant}(\text{Quant}(x))| \leq \frac{y\_scale}{2}$$

This comes from:
1. Rounding error: ±0.5 quantization steps
2. One quantization step = `y_scale`
3. Maximum error = 0.5 × `y_scale`

**Example:** If `y_scale = 0.01`, maximum error is 0.005 (half a percent).
```
**Source:** Signal processing quantization theory (Wikipedia)

### Anti-Patterns to Avoid
- **Mixing notation systems:** Don't switch between ONNX names (y_scale) and other conventions (S, scale) mid-document
- **Step-by-step derivations:** User decided against this - show the formula directly, explain how it's formed
- **LaTeX-heavy rendering:** User wants markdown format, not complex LaTeX blocks
- **Hardware pseudocode for boundary ops:** User explicitly deferred this - document integer matmul only
- **Incomplete saturation ranges:** Must specify exact ranges for int8, uint8, int16, uint16, int4, uint4

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Math rendering | Custom LaTeX→PNG converter | GitHub native markdown math | Built-in MathJax since May 2022, works in issues/PRs/wikis |
| Formula verification | Manual calculation | ONNX Runtime reference implementation | Official runtime validates formulas match spec |
| Rounding mode | Standard round() | Specify round-to-nearest-even explicitly | ONNX uses banker's rounding, not standard rounding |
| Saturation clipping | Manual if/else | Document exact data type ranges | ONNX spec defines precise ranges for all types |
| Variable naming | Intuitive names (scale, offset) | ONNX spec names (y_scale, y_zero_point) | Consistency with official spec prevents confusion |

**Key insight:** ONNX specification is the authoritative source. Don't create simplified or "improved" versions of formulas - document the exact spec formulas.

## Common Pitfalls

### Pitfall 1: Rounding Mode Ambiguity
**What goes wrong:** Using "round()" without specifying the rounding mode leads to implementation errors
**Why it happens:** Standard rounding (round-half-up) differs from ONNX's round-to-nearest-even
**How to avoid:** Always specify "round-to-nearest-even" or "banker's rounding" explicitly in documentation
**Warning signs:**
- Using generic "round" without qualification
- Not mentioning tie-breaking behavior (0.5 → 0, 1.5 → 2)
- Example calculations that don't show half-value rounding

**Example of correct documentation:**
```markdown
**Rounding:** Division (x / y_scale) uses **round-to-nearest-even** (banker's rounding):
- 0.5 → 0 (rounds to even)
- 1.5 → 2 (rounds to even)
- 2.5 → 2 (rounds to even)
```

### Pitfall 2: Zero-Point Type Confusion
**What goes wrong:** Documentation doesn't clarify that y_zero_point data type determines output type
**Why it happens:** The relationship between zero_point type and output type is implicit in ONNX spec
**How to avoid:** Explicitly state "output y has the same data type as y_zero_point"
**Warning signs:**
- Examples showing uint8 zero_point but int8 output
- No mention of type consistency requirement
- Formulas that don't specify output type determination

### Pitfall 3: Incomplete Saturation Documentation
**What goes wrong:** Only documenting int8/uint8 ranges, omitting int16/int4/uint4
**Why it happens:** Most models use 8-bit, so other types seem unimportant
**How to avoid:** Include complete table of all ONNX-supported quantization types
**Warning signs:**
- Table shows only [-128, 127] and [0, 255]
- No mention of 16-bit or sub-byte quantization
- Examples only use 8-bit values

**Complete saturation range table:**
| Data Type | Range | Bits |
|-----------|-------|------|
| uint2 | [0, 3] | 2 |
| int2 | [-2, 1] | 2 |
| uint4 | [0, 15] | 4 |
| int4 | [-8, 7] | 4 |
| uint8 | [0, 255] | 8 |
| int8 | [-128, 127] | 8 |
| uint16 | [0, 65535] | 16 |
| int16 | [-32768, 32767] | 16 |

### Pitfall 4: Asymmetric vs Symmetric Confusion
**What goes wrong:** Documentation implies symmetric quantization is just asymmetric with zero_point=0
**Why it happens:** Mathematically true, but misses hardware and accuracy implications
**How to avoid:** Document both cases separately with their distinct characteristics
**Warning signs:**
- Only one formula shown with "optional zero_point"
- No discussion of range utilization differences
- Missing explanation of when to use each type

### Pitfall 5: Round-Trip Error Overgeneralization
**What goes wrong:** Stating error bound without conditions (only valid for values within quantization range)
**Why it happens:** Focusing on in-range behavior, forgetting saturation at boundaries
**How to avoid:** Qualify error bounds with "for values within quantization range [r_min, r_max]"
**Warning signs:**
- No mention of range limits
- Error formula doesn't account for saturation
- Examples don't show boundary cases

**Correct documentation:**
```markdown
## Round-Trip Error

For values **within the quantization range** [r_min, r_max]:
$$|x - \text{Dequant}(\text{Quant}(x))| \leq \frac{y\_scale}{2}$$

For values **outside the range** (saturated):
- If x < r_min: error = |x - r_min|
- If x > r_max: error = |x - r_max|
```

### Pitfall 6: GitHub Math Syntax Errors
**What goes wrong:** Formulas don't render because of markdown-LaTeX conflicts
**Why it happens:** Dollar signs clash with markdown, underscores need escaping
**How to avoid:** Use backtick-dollar syntax `` $`...$` `` for inline math with special chars, escape underscores in display math
**Warning signs:**
- Formulas show raw LaTeX in GitHub preview
- Variables with underscores break rendering
- Inline math containing other markdown syntax

**Correct syntax:**
```markdown
<!-- Inline with underscores - use backtick-dollar -->
The scale parameter $`y\_scale`$ determines...

<!-- Display math - escape underscores -->
$$y = \text{saturate}\left(\frac{x}{y\_scale} + y\_zero\_point\right)$$

<!-- Block syntax alternative -->
```math
y = \text{saturate}\left(\frac{x}{y_scale} + y_zero_point\right)
```
```

## Code Examples

Verified patterns from official sources:

### ONNX Specification Formula Notation
```markdown
# QuantizeLinear

**Formula:**
$$y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)$$

**Parameters:**
- `x`: Input tensor (float32, float16, bfloat16, int32)
- `y_scale`: Scale factor (scalar or tensor)
- `y_zero_point`: Zero point (optional, defaults to uint8 value 0)

**Rounding:** Round-to-nearest-even (banker's rounding)
**Saturation:** Clips to output data type range
```
**Source:** https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

### DequantizeLinear Formula
```markdown
# DequantizeLinear

**Formula:**
$$y = (x - x\_zero\_point) \times x\_scale$$

**Parameters:**
- `x`: Quantized input tensor (int8, uint8, int16, uint16, etc.)
- `x_scale`: Scale factor (matches x shape for per-tensor/per-axis/blocked)
- `x_zero_point`: Zero point (optional, defaults to 0)

**Note:** Parameter names use `x_` prefix (not `y_`) because this operates on quantized input
```
**Source:** https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

### GitHub Markdown Math Syntax
```markdown
<!-- Inline math: dollar signs -->
The quantization uses scale $y\_scale$ and zero-point $y\_zero\_point$.

<!-- Inline math: backtick-dollar (preferred when formula has special chars) -->
The error bound is $`\frac{y\_scale}{2}`$ for round-to-nearest rounding.

<!-- Display math: double dollar signs -->
$$
y = \text{saturate}\left(\text{round}\left(\frac{x}{y\_scale}\right) + y\_zero\_point\right)
$$

<!-- Display math: code block with 'math' language -->
```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
```
**Source:** https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

### Numerical Example Pattern
```markdown
## Numerical Example: QuantizeLinear

**Given:**
- Input: x = 2.7
- Scale: y_scale = 0.1
- Zero-point: y_zero_point = 0
- Output type: int8

**Calculation:**
1. Divide: 2.7 / 0.1 = 27.0
2. Round: round(27.0) = 27 (no tie, rounds to nearest integer)
3. Add zero-point: 27 + 0 = 27
4. Saturate: 27 is within [-128, 127], no clipping
5. **Result: y = 27**

**Verification:**
- Dequantize: (27 - 0) × 0.1 = 2.7 ✓ (exact round-trip)
```
**Source:** ONNX Runtime quantization documentation pattern

### Symmetric vs Asymmetric Example
```markdown
## Example: Symmetric vs Asymmetric Quantization

**Scenario:** Quantize weights in range [-1.2, 0.8] to int8

### Symmetric (zero_point = 0)
- Range: max(|-1.2|, |0.8|) = 1.2
- Scale: (2 × 1.2) / 255 = 0.00941
- Effective range: [-1.2, 1.2] (wastes 0.4 of range)
- Values: -1.2 → -127, 0.0 → 0, 0.8 → 85

### Asymmetric (zero_point ≠ 0)
- Scale: (0.8 - (-1.2)) / 255 = 0.00784
- Zero-point: round(-1.2 / 0.00784) = -153
- Effective range: [-1.2, 0.8] (full utilization)
- Values: -1.2 → -153, 0.0 → 0, 0.8 → 102

**Trade-off:** Asymmetric uses range more efficiently but requires zero-point addition in hardware.
```
**Source:** ONNX Runtime quantization concepts

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| External LaTeX rendering (MathJax CDN) | GitHub native markdown math | May 2022 | Formulas render in issues, PRs, wikis without setup |
| Images for equations | LaTeX in markdown | May 2022 | Better accessibility, version control, editability |
| Per-tensor quantization only | Per-tensor, per-axis, and blocked | ONNX opset 13+ | More flexible quantization granularities |
| Float8 without specification | Float8 types in ONNX spec | ONNX 1.14+ (2023) | Standardized low-precision formats |

**Deprecated/outdated:**
- **MathJax external scripts:** GitHub renders natively now, no need for `<script>` tags
- **HTML img tags for equations:** Use markdown math syntax instead
- **Non-standard variable names:** Always use ONNX spec names (y_scale, not S or scale)
- **Round-half-up rounding:** ONNX uses round-to-nearest-even exclusively

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal numerical example complexity**
   - What we know: User wants numerical examples with judgment for clarity
   - What's unclear: Should examples show actual ResNet8 values or simplified pedagogical values?
   - Recommendation: Start with simple values (0.1 scale, small integers), then add section with actual ResNet8 layer values extracted from Phase 9 scripts

2. **Per-axis vs blocked quantization explanation depth**
   - What we know: ONNX supports per-tensor, per-axis, and blocked quantization
   - What's unclear: How deeply to document these variations for boundary operations
   - Recommendation: Briefly mention in QuantizeLinear/DequantizeLinear sections, defer detailed per-axis/blocked discussion to core operations (QLinearConv) where they matter more

3. **Float8 and sub-byte type coverage**
   - What we know: ONNX spec includes uint2, int2, uint4, int4, float8e8m0
   - What's unclear: Whether to document these exotic types in detail
   - Recommendation: Include in saturation range table for completeness, but don't create separate examples (focus on int8/uint8 which the project uses)

## Sources

### Primary (HIGH confidence)
- **ONNX QuantizeLinear specification:** https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
  - Formula, parameters, rounding behavior, saturation ranges
- **ONNX DequantizeLinear specification:** https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html
  - Inverse formula, parameter naming conventions
- **GitHub Mathematical Expressions Guide:** https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
  - Official markdown math syntax, inline vs display, LaTeX support
- **ONNX Runtime Quantization Documentation:** https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
  - Scale calculation formulas (symmetric vs asymmetric), zero-point importance, QDQ format

### Secondary (MEDIUM confidence)
- **Quantization signal processing (Wikipedia):** https://en.wikipedia.org/wiki/Quantization_(signal_processing)
  - Error bound theory (±0.5 quantization step), uniform quantization concepts
- **Round-to-nearest-even research:** Multiple academic sources confirm banker's rounding eliminates bias
  - Used in ONNX spec, verified by reference to Wikipedia rounding article

### Tertiary (LOW confidence)
- **Neural network quantization best practices:** Various blog posts and articles (Medium, NVIDIA blog)
  - General guidance but not specific to ONNX operator documentation
  - Marked as supporting context, not authoritative source

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ONNX spec and GitHub markdown are official, authoritative sources
- Architecture: HIGH - ONNX operator documentation follows clear patterns
- Pitfalls: HIGH - Verified from official docs (rounding mode, saturation ranges, type constraints)
- Formula presentation: HIGH - User decisions clearly specify markdown format, no derivations

**Research date:** 2026-02-02
**Valid until:** 90 days (ONNX spec is stable, GitHub markdown features are mature)

**Key constraints from CONTEXT.md:**
- Format: Markdown (not LaTeX-heavy) ✓
- Explanation style: Describe formula formation, show directly without derivation ✓
- Coverage: Both symmetric and asymmetric cases ✓
- Round-trip relationship: Include section with error bounds ✓
- Variable naming: Use ONNX spec names exactly ✓
- Hardware pseudocode: Skip entirely for boundary operations ✓
- Document structure: Operation-by-operation (QuantizeLinear first, then DequantizeLinear) ✓
