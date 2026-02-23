# Prototype to long-run

## Input

* goal
* prototype_diff (optional)

## Output

* accepted_plan
* implementation_diff
* review_findings

## Steps

1. Build a fast prototype with a cheap/fast model. Treat it as draft, not truth.
2. Human review: keep, drop, and risk-tag each part of the prototype.
3. Freeze acceptance criteria before long-run:
   a. correctness
   b. performance budget
   c. readability/maintainability
   d. test requirements
4. Ask `explorer` for:
   a. impact map
   b. unknowns
   c. one alternative approach that does not depend on the prototype
5. Ask `implementer` for two candidate patches:
   a. prototype-based path
   b. clean-slate path
6. Ask `reviewer` to score both candidates against the fixed criteria and report risks first.
7. Select winner, merge, and run final checks.
8. If anchored quality is suspected, run one "reference-free redesign" pass and compare again.
