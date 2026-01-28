---
quick: 003
type: execute
files_modified:
  - .github/workflows/lint.yml
  - pyproject.toml
autonomous: true

must_haves:
  truths:
    - "CI runs ruff linting on push and pull requests"
    - "Ruff is available as dev dependency via uv"
    - "Linting covers all Python files in scripts/"
  artifacts:
    - path: ".github/workflows/lint.yml"
      provides: "GitHub Actions workflow for ruff linting"
    - path: "pyproject.toml"
      provides: "Ruff dev dependency and configuration"
---

<objective>
Add GitHub Actions CI workflow that runs ruff linting on push and pull requests.

Purpose: Enforce code quality through automated linting in CI pipeline.
Output: Working CI workflow that lints Python code with ruff.
</objective>

<context>
@pyproject.toml (uv package manager, Python >=3.12)

Python files to lint:
- scripts/calibration_utils.py
- scripts/convert.py
- scripts/convert_pytorch.py
- scripts/evaluate.py
- scripts/evaluate_pytorch.py
- scripts/quantize_onnx.py
- scripts/quantize_pytorch.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add ruff as dev dependency with configuration</name>
  <files>pyproject.toml</files>
  <action>
    Add ruff as a dev dependency in pyproject.toml:

    1. Add dev dependency group:
       ```toml
       [dependency-groups]
       dev = ["ruff>=0.8.0"]
       ```

    2. Add basic ruff configuration:
       ```toml
       [tool.ruff]
       target-version = "py312"
       line-length = 88

       [tool.ruff.lint]
       select = ["E", "F", "I", "W"]
       ```

    This configures ruff for Python 3.12, standard line length, and enables:
    - E: pycodestyle errors
    - F: Pyflakes
    - I: isort (import sorting)
    - W: pycodestyle warnings
  </action>
  <verify>Run `uv sync` to verify dependency resolves correctly</verify>
  <done>pyproject.toml contains ruff dev dependency and tool.ruff configuration section</done>
</task>

<task type="auto">
  <name>Task 2: Create GitHub Actions lint workflow</name>
  <files>.github/workflows/lint.yml</files>
  <action>
    Create `.github/workflows/lint.yml` with:

    ```yaml
    name: Lint

    on:
      push:
        branches: [main]
      pull_request:
        branches: [main]

    jobs:
      ruff:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4

          - name: Install uv
            uses: astral-sh/setup-uv@v5

          - name: Set up Python
            run: uv python install 3.12

          - name: Install dependencies
            run: uv sync --group dev

          - name: Run ruff check
            run: uv run ruff check .

          - name: Run ruff format check
            run: uv run ruff format --check .
    ```

    This workflow:
    - Triggers on push to main and PRs targeting main
    - Uses official astral-sh/setup-uv action
    - Installs Python 3.12 via uv
    - Syncs dev dependencies (includes ruff)
    - Runs both ruff check (linting) and ruff format --check (formatting)
  </action>
  <verify>Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('.github/workflows/lint.yml'))"`</verify>
  <done>.github/workflows/lint.yml exists with ruff check and format steps</done>
</task>

<task type="auto">
  <name>Task 3: Fix any existing lint issues</name>
  <files>scripts/*.py</files>
  <action>
    Run ruff locally and fix any issues:

    1. Run `uv run ruff check . --fix` to auto-fix issues
    2. Run `uv run ruff format .` to format code
    3. Review changes and ensure no functional changes

    If there are issues that cannot be auto-fixed, address them manually or add
    appropriate `# noqa` comments with justification.
  </action>
  <verify>
    - `uv run ruff check .` exits with code 0
    - `uv run ruff format --check .` exits with code 0
  </verify>
  <done>All Python files pass ruff linting and formatting checks</done>
</task>

</tasks>

<verification>
1. `uv sync --group dev` completes successfully
2. `uv run ruff check .` passes with no errors
3. `uv run ruff format --check .` passes with no errors
4. `.github/workflows/lint.yml` has valid YAML syntax
</verification>

<success_criteria>
- GitHub Actions workflow file exists at .github/workflows/lint.yml
- Workflow triggers on push and PR to main branch
- Ruff is configured in pyproject.toml as dev dependency
- All existing Python files pass ruff checks
- CI will run ruff check and format verification
</success_criteria>

<output>
After completion, report which files were modified and any lint fixes applied.
</output>
