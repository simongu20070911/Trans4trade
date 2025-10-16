# Artifacts

Binary outputs such as trained weights, serialized pickles, and miscellaneous caches reside here.
The directory is ignored by git (except for this README) to prevent large files from polluting the history.

Default layout:
- `models/weights/` — trained PyTorch checkpoint files.
- `pickles/` — intermediate analytics outputs (`*.pkl`).
- `misc/` — other generated files (e.g., cached notebooks, pycache dumps).
