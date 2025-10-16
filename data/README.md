# Data Directory

This folder hosts raw, intermediate, and processed datasets. Contents are ignored by git to avoid committing large binaries. Expect the following structure:

- `raw/` — raw order book captures and ndjson streams.
- `input_data/` — intermediate CSV outputs used by the preprocessing pipeline.
- `backtest/` — generated results from backtesting scripts.
- `archive/` — historical snapshots retained for reference.
- `experimental/` — scratch data dumps from exploratory runs.
- `scratch/` — temporary files.

Placeholders are kept so collaborators know where to drop inputs locally.
