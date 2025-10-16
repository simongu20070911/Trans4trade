"""Convenient references to project directories."""

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INPUT_DATA_DIR = DATA_DIR / "input_data"
BACKTEST_DIR = DATA_DIR / "backtest"
ARCHIVE_DIR = DATA_DIR / "archive"
EXPERIMENTAL_DATA_DIR = DATA_DIR / "experimental"
SCRATCH_DIR = DATA_DIR / "scratch"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_WEIGHTS_DIR = ARTIFACTS_DIR / "models" / "weights"
PICKLES_DIR = ARTIFACTS_DIR / "pickles"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_directories_exist() -> None:
    """Create well-known directories if they are missing."""

    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        INPUT_DATA_DIR,
        BACKTEST_DIR,
        ARCHIVE_DIR,
        EXPERIMENTAL_DATA_DIR,
        SCRATCH_DIR,
        ARTIFACTS_DIR,
        MODEL_WEIGHTS_DIR,
        PICKLES_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INPUT_DATA_DIR",
    "BACKTEST_DIR",
    "ARCHIVE_DIR",
    "EXPERIMENTAL_DATA_DIR",
    "SCRATCH_DIR",
    "ARTIFACTS_DIR",
    "MODEL_WEIGHTS_DIR",
    "PICKLES_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "ensure_directories_exist",
]
