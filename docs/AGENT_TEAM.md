# Agent Team (Codex Multi-Agent)

This project ships two agent groups:

1. General engineering agents
2. OPSX workflow agents for OpenSpec

## General Agents

- `explorer`: Read-only codebase mapping and impact analysis
- `researcher`: Read-only sourced research for libraries, standards, and options
- `implementer`: Workspace-write patch implementation
- `reviewer`: Read-only regression/test-gap review

## OPSX Agents

- `opsx_explore`: Explore mode (no implementation edits)
- `opsx_new`: Create a new change and show first artifact instructions
- `opsx_continue`: Create the next ready artifact for a change
- `opsx_ff`: Fast-forward and generate all remaining artifacts
- `opsx_apply`: Implement tasks from a change
- `opsx_verify`: Verify implementation against artifacts/tasks
- `opsx_sync`: Sync delta specs into main specs
- `opsx_archive`: Archive one completed change
- `opsx_bulk_archive`: Archive multiple completed changes
- `opsx_onboard`: Guided end-to-end OpenSpec onboarding

## Canonical OPSX Flow

Use this default path for most work:

1. `opsx_explore` (optional) to clarify scope
2. `opsx_new` or `opsx_ff` to set up artifacts
3. `opsx_continue` until artifact graph is complete (skip if `opsx_ff`)
4. `opsx_apply` to implement tasks
5. `opsx_verify` to validate implementation
6. `opsx_sync` if delta specs must be merged
7. `opsx_archive` to finalize change

## Paper-Driven Flow (Research-to-OPSX)

Use this flow when scope or acceptance criteria depend on external NN papers.

1. `researcher`: produce a sourced paper brief (claims, metrics, protocol, reproducibility notes, unknowns)
2. `opsx_new` or `opsx_ff`: create OpenSpec change artifacts from that brief
3. `opsx_continue`: finish remaining artifacts so tasks are implementation-ready
4. `opsx_apply`: implement tasks in code
5. `opsx_verify`: validate both artifact completion and paper-backed constraints
6. `opsx_sync` and `opsx_archive`: finalize as needed

Paper brief handoff checklist:
- Source links for every key claim and number
- Evaluation constraints to preserve (dataset/split/metric/baselines)
- Reproducibility constraints (critical hyperparameters, seeds, calibration details)
- Ranked experiments and expected outcomes

## Command-to-Agent Mapping

If you use OPSX-style command names, map them as follows:

- `/opsx:explore` -> `opsx_explore`
- `/opsx:new` -> `opsx_new`
- `/opsx:continue` -> `opsx_continue`
- `/opsx:ff` -> `opsx_ff`
- `/opsx:apply` -> `opsx_apply`
- `/opsx:verify` -> `opsx_verify`
- `/opsx:sync` -> `opsx_sync`
- `/opsx:archive` -> `opsx_archive`
- `/opsx:bulk-archive` -> `opsx_bulk_archive`
- `/opsx:onboard` -> `opsx_onboard`

## Notes

- OPSX agents use the matching `.codex/skills/openspec-*/SKILL.md` file as their workflow source of truth.
- `paper-driven flow` is separate from canonical OPSX flow, but implementation is still done through OPSX (`opsx_apply` + `opsx_verify`).
- Use `opsx_bulk_archive` when archiving several completed changes in parallel.
- Use `opsx_onboard` for a narrated first run; use the other OPSX agents for normal day-to-day flow.
