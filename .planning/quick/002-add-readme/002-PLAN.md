---
type: quick
task: 002-add-readme
description: Add README.md documenting project purpose, structure, and usage
files_modified: [README.md]
autonomous: true
---

<objective>
Create a comprehensive README.md for the ResNet8 CIFAR-10 project.

Purpose: Enable users to understand the project purpose, navigate the codebase, and use the conversion/quantization tools effectively.

Output: A well-structured README.md at the project root.
</objective>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@docs/QUANTIZATION_ANALYSIS.md
@pyproject.toml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create README.md</name>
  <files>README.md</files>
  <action>
Create README.md with the following structure:

1. **Title and Overview**
   - ResNet8 CIFAR-10 Model Conversion and Quantization
   - Brief description: Multi-framework model conversion from Keras to ONNX and PyTorch, with PTQ support

2. **Project Purpose**
   - Source: MLCommons TinyMLPerf ResNet8 model
   - Goal: Cross-framework inference with >85% accuracy
   - Quantization: int8/uint8 static PTQ for model compression

3. **Quick Results Table**
   Use data from STATE.md:
   | Model | Accuracy | Size |
   |-------|----------|------|
   | FP32 baseline | 87.19% | 315KB |
   | ONNX Runtime uint8 | 86.75% | 123KB |
   | ONNX Runtime int8 | 85.58% | 123KB |
   | PyTorch int8 | 85.68% | 165KB |

4. **Project Structure**
   ```
   resnet8/
   ├── scripts/           # Conversion and evaluation scripts
   │   ├── convert.py          # Keras → ONNX
   │   ├── convert_pytorch.py  # ONNX → PyTorch
   │   ├── evaluate.py         # ONNX model evaluation
   │   ├── evaluate_pytorch.py # PyTorch model evaluation
   │   ├── quantize_onnx.py    # ONNX Runtime PTQ
   │   ├── quantize_pytorch.py # PyTorch PTQ
   │   └── calibration_utils.py # Calibration data utilities
   ├── models/            # Converted and quantized models
   ├── docs/              # Analysis documents
   └── logs/              # Conversion and evaluation logs
   ```

5. **Prerequisites**
   - Python 3.12+
   - uv package manager (recommended)
   - CIFAR-10 dataset path
   - Original Keras model path (for conversion)

6. **Installation**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

7. **Usage Examples**
   Show actual commands for each script:

   **Convert Keras to ONNX:**
   ```bash
   uv run python scripts/convert.py /path/to/model.h5 models/resnet8.onnx
   ```

   **Evaluate ONNX model:**
   ```bash
   uv run python scripts/evaluate.py models/resnet8.onnx /path/to/cifar-10-batches-py
   ```

   **Quantize with ONNX Runtime:**
   ```bash
   uv run python scripts/quantize_onnx.py models/resnet8.onnx /path/to/cifar-10-batches-py --type uint8
   ```

   **Convert to PyTorch and quantize:**
   ```bash
   uv run python scripts/convert_pytorch.py models/resnet8.onnx models/resnet8.pt
   uv run python scripts/quantize_pytorch.py models/resnet8.pt /path/to/cifar-10-batches-py
   ```

8. **Architecture**
   - ResNet8 with 3 stacks (16→32→64 filters)
   - Input: 32x32x3 CIFAR-10 images
   - Output: 10-class softmax
   - Reference to docs/QUANTIZATION_ANALYSIS.md for detailed analysis

9. **License** (if applicable) or omit

Keep the README concise and practical - focus on getting users productive quickly.
  </action>
  <verify>
    - File exists: README.md
    - Contains all sections: overview, structure, installation, usage
    - Code blocks are properly formatted
    - Links work (if any internal links)
  </verify>
  <done>
    README.md exists at project root with complete documentation of project purpose, structure, installation, and usage examples for all scripts.
  </done>
</task>

</tasks>

<verification>
- [ ] README.md exists at project root
- [ ] All script names and paths are accurate
- [ ] Usage examples reflect actual script interfaces
- [ ] Results table matches STATE.md data
</verification>

<success_criteria>
- README.md provides clear project overview
- Users can understand purpose and navigate structure
- Usage examples are copy-pasteable and accurate
</success_criteria>

<output>
After completion, create `.planning/quick/002-add-readme/002-SUMMARY.md`
</output>
