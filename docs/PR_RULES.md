# PR Rules

## Purpose
This document defines how code changes should be proposed and reviewed, even if development is done solo with AI assistance.

## Before making a change
Always answer these questions first:
1. What problem is being solved?
2. Which files should change?
3. Does this affect data format or split logic?
4. Could this introduce leakage?
5. How will the change be tested?

## PR size rules
- Keep each PR focused on one purpose.
- Avoid mixing dataset refactors with model experiments in the same PR.
- Avoid mixing feature engineering and evaluation changes in the same PR.

## Recommended PR categories
- `data:` loading / reshaping / cleaning
- `split:` train/val/test logic
- `features:` feature extraction
- `model:` baseline or training updates
- `eval:` metrics and evaluation
- `docs:` documentation and prompt/context changes

## PR checklist
Every PR should include:

- summary of the change
- why the change is needed
- files modified
- risks
- how it was tested
- what outputs were produced

## Example PR template
### Summary
Short description of the change.

### Why
Why this change is needed.

### Files changed
- `src/...`
- `scripts/...`

### Risks
Possible breakage, leakage, or format changes.

### Testing
Commands run:
- `python ...`
- output checked:
  - file exists
  - row counts
  - columns
  - metric output

### Result
What changed after the PR.

## Hard review rules
Reject or revise a PR if:
- it changes split logic without justification
- it changes label logic without inspection output
- it introduces hidden assumptions
- it couples scripts too tightly
- it makes reproduction harder