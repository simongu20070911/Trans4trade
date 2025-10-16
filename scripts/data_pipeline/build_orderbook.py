"""Expand Binance depth snapshots into tabular order book CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from natsort import natsorted

from trans4trade.paths import INPUT_DATA_DIR, RAW_DATA_DIR


def get_orderbook_dict(filepath: Path) -> dict[float, dict[str, object]]:
    orderbook: dict[float, dict[str, object]] = {}
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line.replace("'", '"'))
            if "lastUpdateId" not in record:
                continue
            timestamp = float(record["datetime"])
            record["datetime"] = timestamp
            orderbook[timestamp] = record
    return orderbook


def expand_orderbook(orderbook: dict[float, dict[str, object]], levels: int = 10) -> pd.DataFrame:
    records = len(orderbook)
    datetime_arr = np.zeros(records, dtype="float64")
    last_update_ids = np.zeros(records, dtype="int64")
    bids = np.zeros((records, levels * 2))
    asks = np.zeros((records, levels * 2))

    for row_idx, (timestamp, value) in enumerate(orderbook.items()):
        datetime_arr[row_idx] = timestamp
        last_update_ids[row_idx] = int(value["lastUpdateId"])
        for level_idx, (price, qty) in enumerate(value["bids"][:levels]):
            bids[row_idx, 2 * level_idx] = float(price)
            bids[row_idx, 2 * level_idx + 1] = float(qty)
        for level_idx, (price, qty) in enumerate(value["asks"][:levels]):
            asks[row_idx, 2 * level_idx] = float(price)
            asks[row_idx, 2 * level_idx + 1] = float(qty)

    result: dict[str, object] = {
        "datetime": datetime_arr,
        "lastUpdateId": last_update_ids,
    }
    for level in range(levels):
        result[f"ask{level + 1}"] = asks[:, 2 * level]
        result[f"askqty{level + 1}"] = asks[:, 2 * level + 1]
        result[f"bid{level + 1}"] = bids[:, 2 * level]
        result[f"bidqty{level + 1}"] = bids[:, 2 * level + 1]

    return pd.DataFrame(result)


def process_file(filepath: Path, levels: int) -> pd.DataFrame:
    orderbook = get_orderbook_dict(filepath)
    return expand_orderbook(orderbook, levels=levels)


def build_orderbook_for_date(date: str, *, levels: int, raw_dir: Path, output_dir: Path) -> Path:
    destination_dir = output_dir / date
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "orderbook.csv"

    if destination.exists():
        print(f"Skipping {date}; output already exists at {destination}")
        return destination

    pattern = f"orderbook*"
    files_to_process = natsorted((raw_dir / date).glob(pattern))
    if not files_to_process:
        raise FileNotFoundError(f"No raw orderbook files found for {date} under {raw_dir}")

    frames = [process_file(path, levels) for path in files_to_process]
    combined = pd.concat(frames, axis=0).sort_values(by="datetime")
    combined.to_csv(destination, index=False)
    print(f"Wrote expanded orderbook to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="Trading dates to process (format: DD-Mmm-YYYY).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of order book levels to retain.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw ndjson orderbook snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INPUT_DATA_DIR,
        help="Directory where expanded CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for date in args.dates:
        build_orderbook_for_date(
            date,
            levels=args.levels,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
