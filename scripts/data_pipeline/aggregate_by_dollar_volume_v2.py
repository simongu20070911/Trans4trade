"""Extended aggregation utilities with optional diagnostics plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trans4trade.paths import FIGURES_DIR, INPUT_DATA_DIR
from trans4trade.utils import cleaner


DEFAULT_DATES = [
    "23-Aug-2024",
    "24-Aug-2024",
    "25-Aug-2024",
    "26-Aug-2024",
    "27-Aug-2024",
    "29-Aug-2024",
    "30-Aug-2024",
    "31-Aug-2024",
    "01-Sep-2024",
    "02-Sep-2024",
    "03-Sep-2024",
    "04-Sep-2024",
    "05-Sep-2024",
    "06-Sep-2024",
    "07-Sep-2024",
]


def load_snapshots(dates: Iterable[str]) -> list[pd.DataFrame]:
    frames = []
    for date in dates:
        csv_path = INPUT_DATA_DIR / date / "orderbook_agg_trade.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing aggregated orderbook for {date}: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return frames


def describe_snapshots(dates: Iterable[str], frames: Iterable[pd.DataFrame]) -> None:
    for date, df in zip(dates, frames):
        dollar_volume = df["price"] * df["quantity"]
        print(
            "| date: %s | mean price: %.2f ± %.2f | mean qty %.5f ± %.5f | "
            "mean dollar vol %.2f ± %.2f |"
            % (
                date,
                df["price"].mean(),
                df["price"].std(),
                df["quantity"].mean(),
                df["quantity"].std(),
                dollar_volume.mean(),
                dollar_volume.std(),
            )
        )


def plot_random_windows(frames: list[pd.DataFrame], *, window: int = 5000) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for ax, df in zip(axes.flatten(), frames[:9]):
        start = np.random.randint(0, max(len(df) - window, 1))
        ax2 = ax.twinx()
        ax.plot(df["price"].iloc[start : start + window], label="price")
        ax2.plot(df["quantity"].iloc[start : start + window], color="orange", label="quantity")
        ax.set_xlabel("tick")
        ax.set_ylabel("price")
        ax2.set_ylabel("quantity")
    fig.suptitle("Random windows of price and quantity")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES_DIR / "orderbook_windows.png"
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def aggregate(
    dates: Iterable[str],
    output_folder: str,
    *,
    target_dollar_volume: float | None = None,
    plot_windows: bool = False,
) -> Path:
    frames = load_snapshots(dates)
    describe_snapshots(dates, frames)

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

    target = target_dollar_volume or combined["dollarvol"].mean()
    aggregated = cleaner.group_book_by_dollarvol2(combined, column_map, target)

    dest_dir = INPUT_DATA_DIR / output_folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / "orderbook_agg_trade_dollarvol.csv"
    aggregated.to_csv(output_path, index=False)

    if plot_windows:
        figure_path = plot_random_windows(frames)
        print(f"Saved preview figure to {figure_path}")

    print(f"Wrote aggregated dataset to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="List of trading dates to aggregate (format: DD-Mmm-YYYY).",
    )
    parser.add_argument(
        "--output-folder",
        default="All_to_Sept",
        help="Subfolder within data/input_data where the aggregated CSV will be saved.",
    )
    parser.add_argument(
        "--target-dollar-volume",
        type=float,
        default=None,
        help="Override the automatic mean dollar volume used for grouping.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save preview plots of random price/quantity windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        args.dates,
        args.output_folder,
        target_dollar_volume=args.target_dollar_volume,
        plot_windows=args.plot,
    )


if __name__ == "__main__":
    main()
