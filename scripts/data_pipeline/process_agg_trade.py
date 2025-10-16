"""Process aggTrade streams into tidy CSVs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from natsort import natsorted

from trans4trade.paths import INPUT_DATA_DIR, RAW_DATA_DIR


def get_aggtrade_dict(filepath: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line.replace("'", '"'))
            if "aggTrade" not in line:
                continue
            record["datetime"] = datetime.fromtimestamp(float(record["datetime"]))
            records.append(record)
    return records


def expand_aggtrade(filepath: Path) -> pd.DataFrame:
    return pd.DataFrame(get_aggtrade_dict(filepath))


def process_date(date: str, *, raw_dir: Path, output_dir: Path) -> Path:
    destination_dir = output_dir / date
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "aggTrade.csv"

    if destination.exists():
        print(f"Skipping {date}; output already exists at {destination}")
        return destination

    files = natsorted((raw_dir / date).glob("orderbook*"))
    if not files:
        raise FileNotFoundError(f"No raw orderbook files found for {date} under {raw_dir}")

    frames = [expand_aggtrade(path) for path in files]
    combined = pd.concat(frames, axis=0)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined.sort_values("datetime", inplace=True)
    combined.to_csv(destination, index=False)
    print(f"Wrote aggTrade CSV to {destination}")
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
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw ndjson files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INPUT_DATA_DIR,
        help="Directory where processed CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for date in args.dates:
        process_date(date, raw_dir=args.raw_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
