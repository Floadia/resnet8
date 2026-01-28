---
phase: quick
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - requirements.txt
  - .gitignore
  - .python-version
autonomous: true

must_haves:
  truths:
    - "uv sync installs all dependencies successfully"
    - "Python scripts run correctly with uv run"
    - "Project uses pyproject.toml as single source of dependency truth"
  artifacts:
    - path: "pyproject.toml"
      provides: "uv-compatible project configuration with all dependencies"
      contains: "[project]"
    - path: ".python-version"
      provides: "Pinned Python version for uv"
      contains: "3.12"
  key_links:
    - from: "pyproject.toml"
      to: "requirements.txt dependencies"
      via: "migrated dependencies"
      pattern: "dependencies.*="
---

<objective>
Migrate from pip/requirements.txt to uv for Python package management.

Purpose: uv is a faster, more reliable Python package manager written in Rust. It replaces pip, venv, and pip-tools with a single unified tool.

Output: pyproject.toml with all dependencies, .python-version file, updated .gitignore for uv artifacts
</objective>

<execution_context>
@/home/impactaky/shelffiles/config/claude/get-shit-done/workflows/execute-plan.md
@/home/impactaky/shelffiles/config/claude/get-shit-done/templates/summary.md
</execution_context>

<context>
Current setup:
- requirements.txt exists with pip-style dependencies including git+https source
- No pyproject.toml
- .gitignore has venv/ patterns
- Python scripts in scripts/ directory
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create pyproject.toml with uv configuration</name>
  <files>pyproject.toml, .python-version</files>
  <action>
Create pyproject.toml with:
- [project] section with name="resnet8-cifar10", version="1.2.0", requires-python=">=3.12"
- Migrate all dependencies from requirements.txt:
  - tensorflow>=2.20.0
  - onnx>=1.17.0
  - tf2onnx from git (use uv syntax: "tf2onnx @ git+https://github.com/onnx/tensorflow-onnx.git")
  - numpy>=1.26.4,<2.0
  - protobuf>=5.28.0
  - onnxruntime>=1.23.2
  - torch>=2.0.0
  - torchvision>=0.15.0
  - onnx2torch>=1.5.15
- Add [tool.uv] section if needed for any special configuration

Create .python-version file with "3.12" to pin Python version for uv.
  </action>
  <verify>cat pyproject.toml shows valid TOML with all dependencies; cat .python-version shows 3.12</verify>
  <done>pyproject.toml contains all migrated dependencies in uv-compatible format</done>
</task>

<task type="auto">
  <name>Task 2: Update .gitignore for uv artifacts</name>
  <files>.gitignore</files>
  <action>
Add uv-specific ignores to .gitignore:
- uv.lock (optional - some projects commit this, but for a small ML project, ignoring is fine)
- .venv/ is already covered by existing venv patterns

Keep existing patterns. Add comment section for uv artifacts.
  </action>
  <verify>grep -q "uv" .gitignore returns match</verify>
  <done>.gitignore includes uv-related patterns</done>
</task>

<task type="auto">
  <name>Task 3: Verify uv installation and sync dependencies</name>
  <files>-</files>
  <action>
Run uv sync to create virtual environment and install all dependencies.

If uv is not installed, the executor should note this and provide install instructions (curl -LsSf https://astral.sh/uv/install.sh | sh).

After sync, verify by running: uv run python -c "import tensorflow; import torch; import onnx; print('All imports OK')"

Note: requirements.txt can be kept for backwards compatibility or removed. Recommend keeping it with a comment that pyproject.toml is the source of truth.
  </action>
  <verify>uv sync completes without error; uv run python -c "import tensorflow; import torch; import onnx; print('OK')" prints OK</verify>
  <done>uv environment works, all dependencies install correctly</done>
</task>

</tasks>

<verification>
1. `uv sync` creates .venv and installs all packages
2. `uv run python scripts/evaluate.py --help` works (proves scripts can run with uv)
3. pyproject.toml is valid TOML and contains all original dependencies
</verification>

<success_criteria>
- pyproject.toml exists with all dependencies from requirements.txt
- .python-version pins Python 3.12
- uv sync installs all dependencies without errors
- Scripts can be run via `uv run python scripts/<script>.py`
</success_criteria>

<output>
After completion, create `.planning/quick/001-use-uv-instead-of-old-python-management-/001-SUMMARY.md`
</output>
