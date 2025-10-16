"""Stream Binance order book and aggregate trade data to ndjson files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import sleep, time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import websocket

from trans4trade.paths import RAW_DATA_DIR, ensure_directories_exist


@dataclass
class ListenerState:
    output_root: Path
    interruption_log: List[Dict[str, Optional[datetime]]] = field(default_factory=list)

    def daily_dir(self) -> Path:
        directory = self.output_root / datetime.now().strftime("%d-%b-%Y")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def report_dir(self) -> Path:
        directory = (
            self.output_root
            / "BTCUSDT"
            / "@depth10@100ms"
            / "reports"
            / datetime.now().strftime("%d-%b-%Y")
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory


STATE: ListenerState | None = None


def log_interruption_start() -> None:
    if STATE is None:
        return
    STATE.interruption_log.append({"start": datetime.now(), "end": None})


def log_interruption_end() -> None:
    if STATE is None or not STATE.interruption_log:
        return
    if STATE.interruption_log[-1]["end"] is None:
        STATE.interruption_log[-1]["end"] = datetime.now()


def on_message(ws: websocket.WebSocketApp, message: str) -> None:  # type: ignore[name-defined]
    try:
        payload = json.loads(message)
        header_count = getattr(ws, "header_count", 0) + 1
        setattr(ws, "header_count", header_count)

        payload["datetime"] = str(time())
        destination_dir = STATE.daily_dir() if STATE else RAW_DATA_DIR
        destination = destination_dir / f"orderbook_ws_{header_count // 100_000}.ndjson"
        with destination.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        print(f"Data written to: {destination}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error in on_message: {exc}")
        restart_script()


def on_open(ws: websocket.WebSocketApp) -> None:  # type: ignore[name-defined]
    try:
        print("WebSocket connection opened")
        if STATE is None:
            raise RuntimeError("Listener state not initialised")
        STATE.daily_dir()
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@aggTrade", "btcusdt@depth10@100ms"],
            "id": 1,
        }
        ws.send(json.dumps(subscribe_message))
        print(f"Subscribed to: {subscribe_message['params']}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error in on_open: {exc}")
        restart_script()


def on_close(ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:  # type: ignore[name-defined]
    print(f"WebSocket closed with status: {close_status_code}, message: {close_msg}")
    log_interruption_start()
    restart_script()


def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:  # type: ignore[name-defined]
    print(f"WebSocket error: {error}")
    log_interruption_start()
    restart_script()


def restart_script() -> None:
    print("Restarting script...")
    sleep(1)
    python = sys.executable
    websocket.enableTrace(False)
    os.execl(python, python, *sys.argv)


def generate_report() -> None:
    if STATE is None:
        return
    report_dir = STATE.report_dir()
    report_file = report_dir / "downtime_report.txt"

    with report_file.open("w", encoding="utf-8") as handle:
        for interruption in STATE.interruption_log:
            start = interruption["start"].strftime("%H:%M:%S")
            end_dt = interruption.get("end")
            end = end_dt.strftime("%H:%M:%S") if end_dt else "Ongoing"
            duration = (
                (end_dt - interruption["start"]).total_seconds() if end_dt else "N/A"
            )
            handle.write(f"Interruption from {start} to {end}, Duration: {duration} seconds\n")

    downtime_intervals = [
        (entry["start"], entry["end"]) for entry in STATE.interruption_log if entry.get("end")
    ]
    if not downtime_intervals:
        print(f"No downtime recorded. Report stored at {report_file}")
        return

    start_times = [start.timestamp() for start, _ in downtime_intervals]
    end_times = [end.timestamp() for _, end in downtime_intervals]

    plt.figure(figsize=(10, 6))
    plt.hlines(1, min(start_times), max(end_times), colors="green", label="Uptime")
    plt.hlines(1, start_times, end_times, colors="red", lw=6, label="Downtime")
    plt.xlabel("Timestamp")
    plt.ylabel("Status")
    plt.title("WebSocket Connection Downtime")
    plt.legend()
    plt.grid(True)

    figure_path = report_dir / "downtime_visualization.png"
    plt.savefig(figure_path)
    plt.close()
    print(f"Report and visualization generated at: {report_dir}")


def start_websocket() -> None:
    socket = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(
        socket,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error,
    )
    try:
        ws.run_forever()
    except Exception as exc:  # noqa: BLE001
        print(f"WebSocket run_forever error: {exc}")
        restart_script()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory where streamed ndjson files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories_exist()
    args = parse_args()
    global STATE
    STATE = ListenerState(args.output_root)
    try:
        start_websocket()
    finally:
        log_interruption_end()
        generate_report()


if __name__ == "__main__":
    main()
