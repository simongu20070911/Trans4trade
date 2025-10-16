"""Visualise backtesting performance against forecasting horizon."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BACKTEST_RESULTS = pd.DataFrame(
    {
        "forecasting_horizon": list(range(2, 32)),
        "highest_r2_score": [
            0.3490,
            0.3681,
            0.3754,
            0.3673,
            0.3618,
            0.3486,
            0.3393,
            0.3249,
            0.3173,
            0.3042,
            0.2938,
            0.2793,
            0.2688,
            0.2595,
            0.2544,
            0.2437,
            0.2362,
            0.2252,
            0.2196,
            0.2171,
            0.2182,
            0.2018,
            0.1885,
            0.1871,
            0.2362,
            0.1785,
            0.1698,
            0.1599,
            0.1615,
            0.1615,
        ],
    }
)


def analyze(df: pd.DataFrame) -> None:
    print("Descriptive Statistics:")
    print(df.describe())

    anomaly = df[df["forecasting_horizon"] == 26]
    if not anomaly.empty:
        print("\nPotential anomaly at Horizon 26:")
        print(anomaly)

    correlation = df["forecasting_horizon"].corr(df["highest_r2_score"])
    print(f"\nCorrelation between Forecasting Horizon and R² Score: {correlation:.4f}")


def plot(df: pd.DataFrame, *, output: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df["forecasting_horizon"],
        df["highest_r2_score"],
        marker="o",
        color="tab:blue",
        linestyle="-",
        markersize=6,
    )
    ax.set_title("Highest R² Score vs. Forecasting Horizon")
    ax.set_xlabel("Forecasting Horizon")
    ax.set_ylabel("Highest R² Score")
    ax.grid(True)
    ax.set_xticks(df["forecasting_horizon"])
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the plot instead of showing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(BACKTEST_RESULTS)
    plot(BACKTEST_RESULTS, output=args.save)


if __name__ == "__main__":
    main()
