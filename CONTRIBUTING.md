Contributing Guide

Thanks for your interest in contributing! This repo is an active research workspace, so the focus is on clarity, reproducibility, and safe iteration. Please follow the guidelines below to keep things tidy and professional.

Environment
- Use Python 3.9+ in a virtual environment.
- Install dev tools: `pip install -r requirements-dev.txt` or `make install-dev`.

Code style
- Format with Black and isort: `make format`.
- Lint with Ruff: `make lint` (or `make lint-fix` to auto-fix simple issues).
- Aim for small, focused changes. Prefer explicit names over abbreviations.

Notebooks
- Keep notebooks focused; add brief markdown cells describing goals and key findings.
- Avoid committing large outputs. If necessary, clear outputs before commit.
- Place large datasets under ignored folders (see `.gitignore`).

Pull requests
- Include: a short description, context, and any data assumptions.
- Run `make check` locally before opening a PR.
- If adding new scripts, include a short usage example in the PR or `README.md`.

Data & secrets
- Do not commit secrets or credentials. Use `.env` (ignored); see `env.example`.
- Store large data under `data/` (e.g., `data/raw`, `data/input_data`), which is ignored by git.

Licensing
- The repo includes `LICENSE`. If adding third-party code, include proper notices and references.
