---
quick: 004
type: execute
files_modified:
  - deno.json
  - docs/api/mod.ts
  - docs/api/scripts.ts
  - docs/api/playground.ts
  - .gitignore
autonomous: true

must_haves:
  truths:
    - "deno doc --html generates browsable API documentation"
    - "Documentation covers scripts/ and playground/ Python APIs"
    - "deno task doc produces HTML output in docs/api/_site/"
  artifacts:
    - path: "deno.json"
      provides: "Deno configuration with doc task"
    - path: "docs/api/mod.ts"
      provides: "Root module re-exporting all API documentation"
    - path: "docs/api/scripts.ts"
      provides: "TypeScript type definitions documenting scripts/ Python API"
    - path: "docs/api/playground.ts"
      provides: "TypeScript type definitions documenting playground/ Python API"
---

<objective>
Add `deno doc` support to generate browsable HTML API documentation from TypeScript type definition files that mirror the Python API surface.

Purpose: Enable `deno doc --html` to generate a browsable documentation site for the project's Python APIs (scripts and playground utilities), providing a developer-friendly reference without requiring Python tooling.
Output: `deno.json` with doc task, TypeScript modules with JSDoc describing all public Python functions, and working `deno task doc` command.
</objective>

<context>
@pyproject.toml (project metadata: resnet8-cifar10 v1.2.0)
@README.md (project structure and usage)

Python API surface to document:
- scripts/convert.py - Keras to ONNX conversion
- scripts/convert_pytorch.py - ONNX to PyTorch conversion
- scripts/evaluate.py - ONNX model evaluation on CIFAR-10
- scripts/evaluate_pytorch.py - PyTorch model evaluation on CIFAR-10
- scripts/quantize_onnx.py - ONNX Runtime static quantization (int8/uint8)
- scripts/quantize_pytorch.py - PyTorch static quantization (int8)
- scripts/calibration_utils.py - Calibration data loading utilities
- scripts/extract_operations.py - ONNX operation extraction
- scripts/validate_qlinearconv.py - QLinearConv validation
- scripts/validate_qlinearmatmul.py - QLinearMatMul validation
- scripts/annotate_qdq_graph.py - QDQ graph annotation
- scripts/visualize_graph.py - ONNX graph visualization
- playground/utils/model_loader.py - Cached model loading (load_onnx_model, load_pytorch_model, load_model_variants, get_model_summary)
- playground/utils/layer_inspector.py - Layer inspection (get_onnx_layer_names, get_pytorch_layer_names, get_all_layer_names, get_layer_type)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create deno.json and TypeScript API documentation modules</name>
  <files>deno.json, docs/api/mod.ts, docs/api/scripts.ts, docs/api/playground.ts</files>
  <action>
    1. Read ALL Python files in scripts/ and playground/utils/ to extract:
       - Module docstrings
       - Function signatures (names, parameters with types, return types)
       - Function docstrings (Args, Returns, Raises sections)
       - CLI argument definitions (argparse) for script entry points

    2. Create `deno.json` at project root:
       ```json
       {
         "name": "resnet8-cifar10",
         "version": "1.2.0",
         "tasks": {
           "doc": "deno doc --html --name='ResNet8 CIFAR-10' --output=docs/api/_site docs/api/mod.ts"
         },
         "compilerOptions": {
           "strict": true
         }
       }
       ```

    3. Create `docs/api/mod.ts` as root module that re-exports from scripts.ts and playground.ts.
       Add a top-level JSDoc module comment describing the project.

    4. Create `docs/api/scripts.ts` with TypeScript type definitions and JSDoc for each script module.
       - Group functions by script file using JSDoc @module tags or namespace-like organization
       - For each public function: mirror the Python signature as a TypeScript function declaration (export function)
       - Use TypeScript types that map to Python types: np.ndarray -> Float32Array | number[][], str -> string, Path -> string, etc.
       - Copy Python docstrings into JSDoc format: Args -> @param, Returns -> @returns, Raises -> @throws
       - For CLI scripts (those with argparse), document the CLI interface in JSDoc @example blocks showing usage
       - Mark functions as declarations only (no implementation needed, just type signatures with JSDoc)

    5. Create `docs/api/playground.ts` with TypeScript type definitions and JSDoc for playground utilities.
       - Document model_loader functions: load_onnx_model, load_pytorch_model, load_model_variants, get_model_summary
       - Document layer_inspector functions: get_onnx_layer_names, get_pytorch_layer_names, get_all_layer_names, get_layer_type
       - Include interfaces for return types (ModelVariants, ModelSummary, LayerInfo, etc.)

    Important: These TypeScript files are documentation-only -- they describe the Python API in TypeScript types for `deno doc` rendering. They are NOT executable code. Use `export function name(...): ReturnType { throw new Error("Python API"); }` pattern so deno doc can parse them (deno doc needs actual function bodies, not just declarations).

    6. Add `docs/api/_site/` to `.gitignore` (generated HTML output should not be committed).
  </action>
  <verify>
    - `deno check docs/api/mod.ts` passes (TypeScript valid)
    - `deno task doc` generates HTML output in docs/api/_site/
    - `ls docs/api/_site/` contains index.html and related assets
    - `deno doc docs/api/mod.ts` outputs documentation to stdout without errors
  </verify>
  <done>
    - deno.json exists with "doc" task configured
    - docs/api/mod.ts, scripts.ts, playground.ts exist with JSDoc
    - `deno task doc` produces browsable HTML documentation
    - All Python public functions from scripts/ and playground/utils/ are documented
    - docs/api/_site/ is in .gitignore
  </done>
</task>

</tasks>

<verification>
1. `deno check docs/api/mod.ts` -- TypeScript compiles without errors
2. `deno doc docs/api/mod.ts` -- outputs documentation to stdout
3. `deno task doc` -- generates HTML in docs/api/_site/
4. Verify docs/api/_site/index.html exists and is non-empty
5. Verify .gitignore contains docs/api/_site/
</verification>

<success_criteria>
- `deno task doc` generates browsable HTML API documentation
- All 12 script files and 2 playground utility modules are documented
- TypeScript type definitions accurately reflect Python function signatures
- Generated HTML site is excluded from git via .gitignore
</success_criteria>

<output>
After completion, report:
- Number of Python functions documented
- Location of generated HTML docs
- How to view the documentation (e.g., `deno task doc && open docs/api/_site/index.html`)
</output>
