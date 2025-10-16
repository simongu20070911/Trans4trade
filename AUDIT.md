Repository Professionalism Audit

Summary
This document highlights areas that can feel unprofessional or brittle and suggests concrete improvements. No files were deleted; these are observations and guidance for future cleanup.

Top issues
- Absolute paths in experiments
  - Several exploratory scripts (e.g., `experiments/experimental_models/all.py`) still hard-code legacy paths. Prefer using `trans4trade.paths` or CLI arguments so they work outside the original workstation.

- Notebook imports
  - Many notebooks still reference `from utils import cleaner` instead of `from trans4trade.utils import cleaner`. Update them gradually to stay aligned with the new package layout.

- Vendored artifacts
  - `src/models_s4/` still contains macOS metadata files (`.DS_Store`, `._*`). Remove them in a follow-up commit; `.gitignore` already blocks re-introduction.

- Dependency manifest
  - Runtime dependencies remain implicit. Capture them in `requirements.txt` or `environment.yml` so others can reproduce results without reverse-engineering imports.

- Testing gap
  - There are no automated tests covering new scripts or the `trans4trade` package. Consider adding smoke tests for data loaders, model factories, and CLI wrappers.

Optional next steps
- Update notebooks to rely on `pip install -e .` (or explicit `sys.path` adjustments) instead of hard-coded `os.chdir` calls.
- Introduce `pre-commit` hooks mirroring CI (black/isort/ruff) for local feedback.
- Add sanity-check CLI commands (e.g., `scripts/data_pipeline/... --help`) to the README.
- Gradually migrate experimental models into structured `src/trans4trade/experiments` modules when they stabilise.
