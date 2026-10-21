# Working on FlashEDA

## Pull requests: one per core change
- Open a **separate PR for each core change**: one module, feature, fix or file
  group (e.g. engine, a writer, the planner, tests for one feature, docs,
  release/version bump). Don't bundle a whole phase into one PR.
- Each PR must pass the test suite on its own.
- When a PR depends on another, base it on that PR's branch (a stack) and say
  so at the top of the description: "Part N of M. Based on #X".
- Keep version bumps, CHANGELOG and README updates in their own PR at the end.
- PR descriptions: short **What**, **Testing**, and how to release if relevant.

## Checks before opening a PR
- `pytest` passes (also on Python 3.9: no 3.10+ syntax, no backslashes in
  f-string expressions).
- Generated code is validated against `Plan.apply()` for pandas, polars and DuckDB.
