"""Combine orderbook snapshots with aggregated trade events."""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from trans4trade.paths import INPUT_DATA_DIR


def load_and_preprocess_data(file_path: Path, timestamp_column: str, *, unit: str | None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df[timestamp_column] = (
        pd.to_datetime(df[timestamp_column], unit=unit) if unit else pd.to_datetime(df[timestamp_column])
    )
    return df.sort_values(timestamp_column)


def process_date(date_str: str, base_dir: Path) -> None:
    """Combine orderbook snapshots and aggTrade events for a single date."""

    dir_path = base_dir / date_str
    orderbook_file = dir_path / "orderbook.csv"
    aggtrade_file = dir_path / "aggTrade.csv"
    output_file = dir_path / "orderbook_agg_trade.csv"

    print(f"Processing date: {date_str}")

    if output_file.exists():
        print(f"Skipped {date_str}: {output_file} already exists.")
        return

    if not (orderbook_file.exists() and aggtrade_file.exists()):
        print(f"Skipped {date_str}: Missing input file(s).")
        return

    print(f"Loading and preprocessing data for {date_str}...")
    orderbook_df = load_and_preprocess_data(orderbook_file, "datetime", unit="s")
    aggtrade_df = load_and_preprocess_data(aggtrade_file, "datetime", unit=None)

    orderbook_df["index_id"] = range(1, len(orderbook_df) + 1)
    total_entries = len(aggtrade_df)
    print(f"Matching and combining data for {date_str}... Total entries: {total_entries}")

    start_time = time.time()
    combined_data = []

    orderbook_idx = 0
    orderbook_len = len(orderbook_df)

    for i, trade_row in aggtrade_df.iterrows():
        event_time = trade_row["datetime"]
        matching_orderbooks = []

        while orderbook_idx < orderbook_len and orderbook_df.iloc[orderbook_idx]["datetime"] <= event_time:
            if orderbook_df.iloc[orderbook_idx]["datetime"] == event_time:
                matching_orderbooks.append(orderbook_df.iloc[orderbook_idx])
            orderbook_idx += 1

        if not matching_orderbooks and orderbook_idx > 0:
            matching_orderbooks.append(orderbook_df.iloc[orderbook_idx - 1])

        for closest_orderbook_row in matching_orderbooks:
            combined_row = {
                "datetime_x": closest_orderbook_row["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                "lastUpdateId": closest_orderbook_row["lastUpdateId"],
                "match_id": closest_orderbook_row["index_id"],
                **{col: closest_orderbook_row[col] for col in closest_orderbook_row.index if col.startswith(("ask", "bid"))},
                "eventtype": trade_row["e"],
                "eventtime": trade_row["E"],
                "symbol": trade_row["s"],
                "sellID": trade_row["a"],
                "price": trade_row["p"],
                "quantity": trade_row["q"],
                "firsttradeID": trade_row["f"],
                "lasttradeID": trade_row["l"],
                "tradetime": trade_row["T"],
                "marketmaker": trade_row["m"],
                "ignore": trade_row["M"],
                "datetime_y": trade_row["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f"),
            }

            combined_data.append(combined_row)

        if i % 100 == 0 and i > 0:
            percent_complete = (i / total_entries) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_entries / (i + 1)
            time_remaining = max(estimated_total_time - elapsed_time, 0)
            time_remaining_str = str(timedelta(seconds=int(time_remaining)))
            print(
                f"Processing {date_str}: {percent_complete:.2f}% complete, Time remaining: {time_remaining_str}",
                end="\r",
            )

    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_file, index=False)
    print(f"\nProcessed {date_str}: Output saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="Trading dates to process (format: DD-Mmm-YYYY)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DATA_DIR,
        help="Directory containing per-date orderbook and aggTrade CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for date_str in args.dates:
        process_date(date_str, args.input_dir)
    print("Processing complete.")
if __name__ == "__main__":
    main()
