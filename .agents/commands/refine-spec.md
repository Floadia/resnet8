# Refine spec

## Input

* spec_file

## Output

## Steps

1. Understand intent from `git diff develop..HEAD <spec_file>`.
2. If scope is unclear, ask `explorer` for impacted files and unknowns.
3. Loop until no gray zones remain:
   a. Update spec while keeping human readability.
   b. For each gray zone, add one `<question>...</question>`.
   c. Ask user to answer the questions.
4. If major edits were made, ask `reviewer` for consistency/regression checks.
5. Ask user for final review and approval.
