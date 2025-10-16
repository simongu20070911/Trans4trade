Trans4trade Codebase(cleaned)

> 📄 **Paper:** The Trans4Trade manuscript lives at `docs/papers/trans4.pdf`. Start there if you need the research narrative or citation details.

- Purpose: Research and experimentation on order book forecasting using a Transformer backbone with State‑Space Models (S4) and linear decoders.
- Highlights: Multiple notebooks exploring preprocessing, normalization, model baselines (LSTM, Transformer, S4), and backtesting. The paper draft is `trans4.pdf`.

What’s in this repo
- Notebooks: curated under `notebooks/` (e.g., `preprocessing/011_data_precleaning.ipynb`, `modeling/112_model_s4_encoder_transformer.ipynb`, `backtesting/214_backtester_transformer_with_s4.ipynb`).
- Python scripts: production-ish utilities live in `scripts/` (e.g., `scripts/data_pipeline/build_orderbook.py`, `scripts/data_pipeline/combine_orderbook_aggregates.py`, `scripts/evaluation/inference_benchmark.py`).
- Models: packaged code under `src/trans4trade/` with vendor S4 modules in `src/models_s4/`.
- Data folders: non-tracked datasets under `data/` (e.g., `data/raw`, `data/input_data`, `data/backtest`).
- Paper: see `docs/papers/trans4.pdf`.

Status
- This is an active research codebase with mixed experiments. CI, linters, editor settings, and contribution guidelines are provided to make it easier to collaborate professionally without moving or deleting any existing files.

Getting started
1) Environment
   - Python 3.9+ recommended.
   - Create a virtual environment and install dev tools:
     - `python -m venv .venv && source .venv/bin/activate`
     - `pip install -U pip`
     - `pip install -r requirements-dev.txt`
     - `pip install -e .` (installs the `trans4trade` package for local imports)

2) Optional runtime dependencies
   - Runtime requirements vary per notebook/script (e.g., `numpy`, `pandas`, `torch`, `matplotlib`, `scikit-learn`, `torchaudio`, `natsort`). Install as needed:
     - `pip install numpy pandas torch matplotlib scikit-learn natsort torchaudio`
   - Consider using CUDA builds of PyTorch when applicable: see pytorch.org for the correct install command for your system.

3) Editor & style
   - Editor settings: `.editorconfig` is provided.
   - Formatting: `black` + `isort`.
   - Linting: `ruff`.
   - Commands:
     - `make format` — format code with black/isort
     - `make lint` — run ruff checks
     - `make lint-fix` — auto-fix simple issues
     - `make check` — CI‑equivalent checks

Data
- By default, data paths referenced in notebooks/scripts point to `data/input_data/`, `data/raw/`, and `data/backtest/`.
- Large data folders and serialized artifacts are ignored for future commits via `.gitignore`.
- If you need environment variables (e.g., for paths or API keys), copy `env.example` to `.env` and edit values.

Typical usage examples
- Precleaning pipeline (adjust dates as needed):
  - `python -m scripts.data_pipeline.build_orderbook --dates 23-Aug-2024 24-Aug-2024`
  - `python -m scripts.data_pipeline.process_agg_trade --dates 23-Aug-2024 24-Aug-2024`
  - `python -m scripts.data_pipeline.combine_orderbook_aggregates --dates 23-Aug-2024 24-Aug-2024`
- Benchmark inference:
  - `python -m scripts.evaluation.inference_benchmark --models pure_attention lstm --runs 100`
- Exploration:
  - Open notebooks in Jupyter: `jupyter lab` or `jupyter notebook`.

Repository layout (high level)
- `notebooks/` — organised experiments (preprocessing, modeling, backtesting, misc).
- `scripts/` — runnable pipeline, evaluation, and legacy utilities.
- `src/trans4trade/` — reusable Python package (helpers, models, paths) with vendor code in `src/models_s4/`.
- `experiments/` — legacy or exploratory assets kept out of the main package.
- `data/` — local-only datasets (ignored by git) described in `data/README.md`.
- `artifacts/` — model weights and serialized outputs (ignored by git).
- `docs/` — papers, structure guides, and translated documentation.
- `reports/` — generated figures and summaries.
- Root configs: `README.md`, `CONTRIBUTING.md`, `.editorconfig`, `.gitattributes`, `pyproject.toml`, `.github/workflows/ci.yml`.

Contributing
- Please read `CONTRIBUTING.md` for coding style, formatting, and PR guidelines.
- Run `make check` before opening a PR.

License
- This repository includes `LICENSE` (all rights reserved). Add a formal license if redistribution is required.

Citation
- See `docs/papers/trans4.pdf` for the accompanying paper/draft.
- Paper: see `docs/papers/trans4.pdf`.
- `docs/` — documentation, guides, and papers (see `docs/papers/trans4.pdf`).
