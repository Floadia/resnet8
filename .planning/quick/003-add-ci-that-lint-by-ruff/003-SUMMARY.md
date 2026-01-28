---
quick: 003
type: summary
subsystem: ci-infrastructure
tags: [ci, github-actions, linting, ruff, code-quality, python]
completed: 2026-01-28
duration: 4min 27s

dependencies:
  requires:
    - quick-001 (uv package manager)
  provides:
    - Automated code quality enforcement via CI
    - Ruff linting and formatting checks
  affects:
    - All future Python code changes (must pass linting)

tech-stack:
  added:
    - ruff>=0.8.0 (linting/formatting)
    - GitHub Actions workflows
  patterns:
    - CI/CD automation
    - Pre-commit style checks in CI

key-files:
  created:
    - .github/workflows/lint.yml
  modified:
    - pyproject.toml
    - scripts/calibration_utils.py
    - scripts/convert.py
    - scripts/convert_pytorch.py
    - scripts/evaluate.py
    - scripts/evaluate_pytorch.py
    - scripts/quantize_onnx.py
    - scripts/quantize_pytorch.py

decisions:
  lint-rules:
    choice: "E (pycodestyle errors), F (Pyflakes), I (isort), W (warnings)"
    reasoning: "Standard baseline rules for Python code quality"
    alternatives: "More extensive rule sets like flake8-bugbear"

  line-length:
    choice: "88 characters"
    reasoning: "Black's default, good balance between readability and compactness"
    alternatives: "79 (PEP 8), 100, 120"

  ci-trigger:
    choice: "Push to main and PRs targeting main"
    reasoning: "Catch issues early in PRs, enforce on main branch"
    alternatives: "All branches, main only"

  python-setup:
    choice: "astral-sh/setup-uv@v5"
    reasoning: "Official uv action, integrates with existing uv setup"
    alternatives: "actions/setup-python + pip install uv"
---

# Quick Task 003: Add CI Linting with Ruff

**One-liner:** GitHub Actions CI workflow enforcing ruff linting (E/F/I/W rules, 88 char lines) on all Python files

## Overview

Added continuous integration workflow to automatically lint Python code using ruff on every push to main and pull request. All existing code updated to pass ruff checks.

## Tasks Completed

### Task 1: Add ruff as dev dependency with configuration
**Commit:** 3e4968b

Added ruff to pyproject.toml dev dependency group with baseline configuration:
- Python 3.12 target
- 88 character line length
- Enabled rules: E (pycodestyle errors), F (Pyflakes), I (isort), W (warnings)

**Files modified:**
- pyproject.toml (added [dependency-groups] and [tool.ruff] sections)

### Task 2: Create GitHub Actions lint workflow
**Commit:** 3c968ad

Created `.github/workflows/lint.yml` workflow that:
- Triggers on push to main branch and PRs targeting main
- Uses astral-sh/setup-uv@v5 for environment setup
- Installs Python 3.12 via uv
- Runs `uv sync --group dev` to install ruff
- Executes `ruff check .` for linting
- Executes `ruff format --check .` for formatting validation

**Files created:**
- .github/workflows/lint.yml

### Task 3: Fix existing lint issues
**Commit:** cc40f18

Fixed all linting violations in existing Python files:
- Auto-fixed 23 issues: import sorting, unused imports, unnecessary f-strings
- Formatted all 7 Python files
- Manually fixed 25 E501 (line too long) violations by splitting long comments and strings
- Added noqa comment for intentional E402 in quantize_pytorch.py (import after sys.path manipulation)

**Files modified:**
- scripts/calibration_utils.py
- scripts/convert.py
- scripts/convert_pytorch.py
- scripts/evaluate.py
- scripts/evaluate_pytorch.py
- scripts/quantize_onnx.py
- scripts/quantize_pytorch.py

## Technical Details

### Ruff Configuration

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

Rule categories enabled:
- **E**: pycodestyle errors (indentation, whitespace, syntax issues)
- **F**: Pyflakes (undefined names, unused imports, etc.)
- **I**: isort (import sorting and organization)
- **W**: pycodestyle warnings (trailing whitespace, etc.)

### CI Workflow Structure

```yaml
jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync --group dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
```

## Lint Fixes Applied

### Auto-fixed Issues (23)
- Import organization (I001): 3 files had unsorted imports
- Unused imports (F401): 1 unused Path import in convert.py
- Unnecessary f-strings (F541): 10 f-strings without placeholders

### Manual Fixes (25)
- **E501 (line too long)**: Split 25 long lines across 7 files
  - Comments split across multiple lines
  - Long strings split with continuation
  - Function signatures reformatted

- **E402 (import not at top)**: Added `# noqa: E402` to calibration_utils import in quantize_pytorch.py (intentional after sys.path manipulation)

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification criteria met:

```bash
$ uv sync --group dev
Resolved 73 packages in 1ms

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
7 files already formatted

$ python3 -c "import yaml; yaml.safe_load(open('.github/workflows/lint.yml'))"
✓ YAML syntax valid
```

## Success Criteria

- [x] GitHub Actions workflow file exists at .github/workflows/lint.yml
- [x] Workflow triggers on push and PR to main branch
- [x] Ruff is configured in pyproject.toml as dev dependency
- [x] All existing Python files pass ruff checks
- [x] CI will run ruff check and format verification

## Impact

### Immediate
- Code quality enforcement on all future changes
- Consistent code style across the project
- Auto-detection of common Python errors (unused imports, undefined names, etc.)

### Future
- Pull requests will fail CI if code doesn't meet quality standards
- Reduced code review burden (automated style checking)
- Foundation for adding more extensive linting rules

## Next Steps

Future enhancements (not in scope of this task):
- Consider adding more ruff rules (e.g., B for bugbear, N for naming)
- Add type checking with mypy
- Add test coverage requirements to CI
- Add security scanning (bandit, safety)

## Metrics

- **Duration:** 4min 27s
- **Files modified:** 8 total (1 created, 1 added config, 7 fixed)
- **Lint fixes:** 48 total (23 auto-fixed, 25 manual)
- **Commits:** 3 atomic commits (dependency, workflow, fixes)

---

*Quick task completed: 2026-01-28*
