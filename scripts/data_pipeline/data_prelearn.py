"""Prepare learning-ready order book dataset."""

from __future__ import annotations

import argparse
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from trans4trade.paths import INPUT_DATA_DIR


def load_orderbooks(dates: Iterable[str]) -> list[pd.DataFrame]:
    frames = []
    for date in dates:
        csv_path = INPUT_DATA_DIR / date / "orderbook.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing orderbook for {date}: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return frames


def add_weighted_mid_price(df: pd.DataFrame, *, price_col: str = "w_midprice") -> pd.DataFrame:
    denominator = df["askqty1"] + df["bidqty1"]
    df[price_col] = (df["ask1"] * df["askqty1"] + df["bid1"] * df["bidqty1"]) / denominator
    return df


def build_learning_dataset(dates: Iterable[str], output_subdir: str) -> pd.DataFrame:
    frames = [add_weighted_mid_price(df.copy()) for df in load_orderbooks(dates)]
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("datetime", inplace=True)

    destination_dir = INPUT_DATA_DIR / output_subdir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "orderbook.csv"
    combined.to_csv(destination, index=False)
    print(f"Wrote learning dataset to {destination}")
    return combined


def maybe_plot(series: pd.Series, *, window: int | None = None) -> None:
    if window:
        subset = series.iloc[:window]
    else:
        subset = series
    plt.figure(figsize=(12, 4))
    plt.plot(subset.to_list())
    plt.title("Weighted mid-price preview")
    plt.xlabel("Tick")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="Trading dates to include (format: DD-Mmm-YYYY).",
    )
    parser.add_argument(
        "--output-subdir",
        default="all",
        help="Sub-directory inside data/input_data for the output CSV.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="If >0, plot the first N weighted mid-price samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_learning_dataset(args.dates, args.output_subdir)
    if args.preview:
        maybe_plot(dataset["w_midprice"], window=args.preview)


if __name__ == "__main__":
    main()

