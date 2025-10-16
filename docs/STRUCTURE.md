Recommended Project Structure (non-disruptive)

The repository has been reorganised to reduce root clutter. The map below reflects the current layout and will serve as the baseline going forward.

Current layout
- `src/trans4trade/`      — installable package (helpers, models, paths)
- `src/models_s4/`        — vendored S4 implementation
- `scripts/`              — CLI entry points (data pipeline, evaluation, legacy)
- `notebooks/`            — organised experiments (preprocessing/modeling/backtesting/misc)
- `experiments/`          — archived exploratory notebooks and models
- `data/`                 — local-only datasets (ignored by git)
- `artifacts/`            — model weights and pickled outputs (ignored by git)
- `reports/`              — generated figures and write-ups
- `docs/`                 — documentation, guides, and papers (`docs/papers/trans4.pdf` is the main manuscript)
- `tests/`                — *(reserved; add when automated tests exist)*

Notes
- Keep all large files and datasets under `data/` (git-ignored) and document assumptions in `data/README.md`.
- Prefer `src/trans4trade/` for importable code; notebooks should import from the package rather than duplicating logic.
- Use `scripts/` for task-oriented entry points and expose sensible `argparse` interfaces.
- Keep notebooks light on outputs for smaller diffs and save important figures to `reports/figures/`.
