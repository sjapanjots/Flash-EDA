# Working on FlashEDA

## Pull requests: one per core change
- Open a **separate PR for each core change**: one module, feature, fix or file
  group (e.g. engine, a writer, the planner, tests for one feature, docs,
  release/version bump). Don't bundle a whole phase into one PR.
- Each PR must pass the test suite on its own.
- **Every PR targets `main`.** When a PR depends on another, branch from that
  PR's branch but still set the base to `main`, and say at the top of the
  description: "Part N of M. Merge after #X". Its diff shrinks to its own
  change once #X is merged. (Basing a PR on another PR's branch caused merges
  to land on side branches instead of `main`.)
- Keep version bumps, CHANGELOG and README updates in their own PR at the end.
- PR descriptions: short **What**, **Testing**, and how to release if relevant.

## Checks before opening a PR
- `pytest` passes (also on Python 3.9: no 3.10+ syntax, no backslashes in
  f-string expressions).
- Generated code is validated against `Plan.apply()` for pandas, polars and DuckDB.
