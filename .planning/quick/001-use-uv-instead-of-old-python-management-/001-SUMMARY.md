---
phase: quick
plan: 001
subsystem: infra
tags: [uv, python, package-management, pyproject.toml]

# Dependency graph
requires: []
provides:
  - uv-based Python package management
  - pyproject.toml as single dependency source
  - .python-version pinned to 3.12
affects: [development-workflow]

# Tech tracking
tech-stack:
  added: [uv, hatchling]
  patterns: [uv-package-management, override-dependencies]

key-files:
  created:
    - pyproject.toml
    - .python-version
  modified:
    - .gitignore
    - requirements.txt

key-decisions:
  - "Use package = false for scripts-only project (no need for installable package)"
  - "Add protobuf override to resolve tf2onnx vs TensorFlow version conflict"
  - "Remove numpy<2.0 upper bound (TensorFlow 2.20+ supports numpy 2.x)"
  - "Keep requirements.txt for backwards compatibility with note about pyproject.toml"

patterns-established:
  - "uv sync for dependency installation"
  - "uv run for script execution"
  - "override-dependencies for transitive dependency conflicts"

# Metrics
duration: 5min
completed: 2026-01-28
---

# Quick Task 001: Use uv Instead of pip Summary

**Migrated to uv package manager with pyproject.toml, resolving tf2onnx/protobuf version conflict via dependency override**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-28T07:52:24Z
- **Completed:** 2026-01-28T07:57:18Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Created pyproject.toml with all dependencies migrated from requirements.txt
- Configured uv for scripts-only project (package = false)
- Resolved protobuf version conflict (tf2onnx needs ~3.20, TensorFlow needs 5.x)
- Verified uv sync installs all 70 packages successfully
- Verified scripts run correctly with `uv run`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pyproject.toml with uv configuration** - `1597382` (chore)
2. **Task 2: Update .gitignore for uv artifacts** - `bd170b0` (chore)
3. **Task 3: Verify uv sync and fix dependency conflicts** - `95707e6` (chore)

## Files Created/Modified
- `pyproject.toml` - uv-compatible project configuration with all dependencies
- `.python-version` - Pins Python version to 3.12 for uv
- `.gitignore` - Added uv.lock to ignored files
- `requirements.txt` - Added note about pyproject.toml as source of truth

## Decisions Made

1. **package = false:** This is a scripts-only project, not an installable Python package. Setting package = false avoids hatchling trying to discover packages to build.

2. **protobuf override:** tf2onnx on GitHub main has `protobuf~=3.20` constraint, but TensorFlow 2.20+ requires protobuf 5.x. Added `override-dependencies = ["protobuf>=5.28.0"]` to resolve this. Works because tf2onnx is compatible with newer protobuf at runtime.

3. **Removed numpy upper bound:** Original requirements.txt had `numpy>=1.26.4,<2.0` but TensorFlow 2.20 with Python 3.13 support requires numpy>=2.1.0. uv's stricter resolver exposed this future incompatibility. Removed upper bound since TensorFlow now supports numpy 2.x.

4. **Keep requirements.txt:** Kept for backwards compatibility with pip users, with a note pointing to pyproject.toml as the source of truth.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Resolved protobuf version conflict**
- **Found during:** Task 3 (uv sync)
- **Issue:** tf2onnx from GitHub requires protobuf~=3.20, but TensorFlow 2.20+ needs protobuf 5.x. uv's strict resolver caught this conflict that pip would silently override.
- **Fix:** Added `override-dependencies = ["protobuf>=5.28.0"]` in [tool.uv] section
- **Files modified:** pyproject.toml
- **Verification:** uv sync completes, imports work
- **Committed in:** 95707e6 (Task 3 commit)

**2. [Rule 3 - Blocking] Resolved numpy version conflict**
- **Found during:** Task 3 (uv sync)
- **Issue:** numpy<2.0 constraint conflicted with ml-dtypes>=0.5.1 (TensorFlow dependency) which requires numpy>=2.1.0 on Python 3.13.
- **Fix:** Removed upper bound, changed to `numpy>=1.26.4`
- **Files modified:** pyproject.toml
- **Verification:** uv sync completes with numpy 2.4.1
- **Committed in:** 95707e6 (Task 3 commit)

**3. [Rule 3 - Blocking] Configured hatchling for git dependencies**
- **Found during:** Task 3 (uv sync)
- **Issue:** hatchling build backend doesn't allow git+https direct references by default
- **Fix:** Added `[tool.hatch.metadata]` with `allow-direct-references = true`
- **Files modified:** pyproject.toml
- **Verification:** Build succeeds
- **Committed in:** 95707e6 (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (all blocking issues)
**Impact on plan:** All auto-fixes necessary for uv to work. uv's stricter dependency resolution exposed real version conflicts that pip would silently override. No scope creep.

## Issues Encountered
- uv is stricter than pip about dependency resolution - this is actually a feature, not a bug. It exposed version conflicts that could cause runtime issues.
- Needed to clean .venv and re-sync to resolve a distutils installation issue with tf2onnx git dependency.

## User Setup Required

None - uv is already installed on this system. For new users:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use:
```bash
uv sync         # Install dependencies
uv run python scripts/evaluate.py  # Run scripts
```

## Next Phase Readiness
- Project now uses modern Python package management
- `uv sync` creates reproducible environments
- Scripts run via `uv run python scripts/*.py`
- uv.lock generated for deterministic builds (currently gitignored)

---
*Plan: quick-001*
*Completed: 2026-01-28*
