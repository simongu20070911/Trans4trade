"""Utility helpers for interacting with the external green-machine project."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from trans4trade.paths import PROJECT_ROOT


DEFAULT_GREEN_MACHINE = PROJECT_ROOT / "data" / "experimental" / "green-machine"


def _resolve_path(env_var: str, default: Path) -> Path:
    override = os.getenv(env_var)
    return Path(override).expanduser().resolve() if override else default.resolve()


def get_order_book_dict() -> None:
    """Append the orderbook utils path from the green-machine project to sys.path."""

    root = _resolve_path("GREEN_MACHINE_PATH", DEFAULT_GREEN_MACHINE)
    sys.path.append(str(root / "system" / "BTCUSDT" / "@depth10@100ms"))


def get_workplace_path() -> None:
    """Append the green-machine project root to sys.path."""

    root = _resolve_path("GREEN_MACHINE_PATH", DEFAULT_GREEN_MACHINE)
    sys.path.append(str(root))
