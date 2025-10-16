"""Aggregate order book data by dollar volume buckets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trans4trade.paths import INPUT_DATA_DIR
from trans4trade.utils import cleaner


DEFAULT_DATES: list[str] = [
    "23-Aug-2024",
    "24-Aug-2024",
    "25-Aug-2024",
    "26-Aug-2024",
    "27-Aug-2024",
    "29-Aug-2024",
]


def load_orderbook_snapshot(date: str) -> pd.DataFrame:
    """Read the aggregated order book CSV for a single date."""

    csv_path = INPUT_DATA_DIR / date / "orderbook_agg_trade.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing aggregated orderbook for {date}: {csv_path}")
    return pd.read_csv(csv_path)


def describe_distribution(date: str, df: pd.DataFrame) -> None:
    """Print quick descriptive statistics for sanity checking."""

    mean_price = df["price"].mean()
    std_price = df["price"].std()
    mean_qty = df["quantity"].mean()
    std_qty = df["quantity"].std()
    dollar_volume = df["price"] * df["quantity"]
    mean_dollarvol = dollar_volume.mean()
    std_dollarvol = dollar_volume.std()

    print(
        "| date: %s | mean price: %.2f ± %.2f | mean qty %.5f ± %.5f | "
        "mean dollar vol %.2f ± %.2f |"
        % (date, mean_price, std_price, mean_qty, std_qty, mean_dollarvol, std_dollarvol)
    )


def aggregate_by_dollar_volume(
    dates: Iterable[str], target_dollar_volume: float = 400.0
) -> tuple[pd.DataFrame, Path]:
    """Aggregate multiple day snapshots into a single, dollar-vol normalized dataset."""

    frames = []
    for date in dates:
        df = load_orderbook_snapshot(date)
        describe_distribution(date, df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["dollarvol"] = combined["quantity"] * combined["price"]

    bid_cols = [f"bid{i}" for i in range(1, 11)] + [f"bidqty{i}" for i in range(1, 11)]
    ask_cols = [f"ask{i}" for i in range(1, 11)] + [f"askqty{i}" for i in range(1, 11)]

    column_map = {
        "price": "price",
        "quantity": "quantity",
        "datetime": "datetime_y",
        "bid_ask_columns": bid_cols + ask_cols,
    }

    aggregated = cleaner.group_book_by_dollarvol2(
        combined, column_map, target_dollar_volume
    )

    dest_dir = INPUT_DATA_DIR / "all"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"orderbook_agg_trade_dollarvol_{int(target_dollar_volume)}.csv"
    aggregated.to_csv(dest_path, index=False)

    print(f"Wrote aggregated dataset to {dest_path}")
    return aggregated, dest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate order book snapshots by dollar volume buckets."
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="List of trading dates to aggregate (format: DD-Mmm-YYYY).",
    )
    parser.add_argument(
        "--target-dollar-volume",
        type=float,
        default=400.0,
        help="Target dollar volume bucket size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_by_dollar_volume(args.dates, args.target_dollar_volume)


if __name__ == "__main__":
    main()

