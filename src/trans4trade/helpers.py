"""Utility helpers for lightweight experiment logging."""

from pathlib import Path
from typing import Iterable, Mapping, Union

import pandas as pd

from .paths import PROJECT_ROOT


DEFAULT_LOG_DIR = PROJECT_ROOT / "experiments" / "experimental_models" / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "experiment.log"

Loggable = Union[pd.DataFrame, Mapping, Iterable, int, float, str]


def save_to_log(data: Loggable, message: str, *, log_file: Path = DEFAULT_LOG_FILE) -> None:
    """Append a short message and arbitrary payload to the experiment log."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n{data}\n\n")


def save_df(data: Union[pd.DataFrame, Iterable], message: str, *, log_file: Path = DEFAULT_LOG_FILE) -> None:
    """Persist tabular data to the experiment log as CSV (append-only)."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")
        dataframe = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        dataframe.to_csv(handle, index=False)
        handle.write("\n")


__all__ = ["save_to_log", "save_df", "DEFAULT_LOG_FILE"]
