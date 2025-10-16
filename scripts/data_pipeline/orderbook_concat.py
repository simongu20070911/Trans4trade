"""Concatenate order book snapshots and compute volume-weighted price."""

from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from trans4trade.paths import INPUT_DATA_DIR

DEFAULT_DATES = [
    "23-Aug-2024",
    "24-Aug-2024",
    "25-Aug-2024",
    "26-Aug-2024",
    "27-Aug-2024",
    "29-Aug-2024",
]


def load_orderbooks(dates: Iterable[str]) -> list[pd.DataFrame]:
    frames = []
    for date in dates:
        csv_path = INPUT_DATA_DIR / date / "orderbook.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing orderbook for {date}: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return frames


def add_volume_weighted_price(df: pd.DataFrame) -> pd.DataFrame:
    denominator = df["askqty1"] + df["bidqty1"]
    df["price"] = (df["ask1"] * df["askqty1"] + df["bid1"] * df["bidqty1"]) / denominator
    return df


def concatenate(dates: Iterable[str], output_subdir: str) -> pd.DataFrame:
    frames = [add_volume_weighted_price(df) for df in load_orderbooks(dates)]
    combined = pd.concat(frames, ignore_index=True)
    combined.set_index("datetime", inplace=True)

    destination_dir = INPUT_DATA_DIR / output_subdir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "orderbook.csv"
    combined.to_csv(destination)
    print(f"Wrote concatenated orderbook to {destination}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="List of trading dates (format: DD-Mmm-YYYY).",
    )
    parser.add_argument(
        "--output-subdir",
        default="All_to_Sept",
        help="Sub-directory inside data/input_data for the combined CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concatenate(args.dates, args.output_subdir)


if __name__ == "__main__":
    main()
