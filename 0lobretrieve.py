import sys
import os
import json
import websocket
import subprocess
from time import time, sleep
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

available = 0
interruption_log = []

# Get the absolute path to the directory containing 'the-green-machine'
green_machine_path = os.path.abspath("/home/gaen/Documents/codespace-gaen/Simons/raw")

# Append the path to 'sys.path'
sys.path.append(green_machine_path)

# Define the WebSocket event handlers
def on_message(ws, message):
    try:
        d = json.loads(message)
        if not hasattr(ws, 'header'):
            ws.header = []
        ws.header.append(0)
        dt = str(time())
        d['datetime'] = dt
        date = datetime.now().strftime('%d-%b-%Y')
        filedir = os.path.join(green_machine_path, f'{date}/')
        if not os.path.exists(filedir):
            os.makedirs(filedir)  # Ensure all intermediate directories are created if they don't exist
        filepath = os.path.join(filedir, f'orderbook_ws_{len(ws.header) // 100_000}.ndjson')
        with open(filepath, mode='a+', encoding='utf-8') as f:
            available = 1
            json.dump(d, f)
            f.write('\n')
        print(f"Data written to: {filepath}")
    except Exception as e:
        print(f"Error in on_message: {e}")
        restart_script()

def on_open(ws):
    try:
        print("WebSocket connection opened")
        date = datetime.now().strftime('%d-%b-%Y')
        filedir = os.path.join(green_machine_path, f'{date}/')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [
                "btcusdt@aggTrade",
                "btcusdt@depth10@100ms"
            ],
            "id": 1
        }
        ws.send(json.dumps(subscribe_message))
        print(f"Subscribed to: {subscribe_message['params']}")
    except Exception as e:
        print(f"Error in on_open: {e}")
        restart_script()

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed with status: {close_status_code}, message: {close_msg}")
    record_interruption()  # Record the interruption time
    restart_script()  # Restart the script

def on_error(ws, error):
    print(f"Error occurred: {error}")
    record_interruption()  # Record the interruption time
    restart_script()

def restart_script():
    print("Restarting script...")
    sleep(1)  # Wait before restarting
    python = sys.executable
    os.execl(python, python, *sys.argv)

def record_interruption():
    # Record the interruption time
    start_time = datetime.now()
    interruption_log.append({'start': start_time, 'end': None})

def end_interruption():
    # Mark the end of the interruption
    if interruption_log and interruption_log[-1]['end'] is None:
        interruption_log[-1]['end'] = datetime.now()

def generate_report():
    # Create a report at the end of the day
    date = datetime.now().strftime('%d-%b-%Y')
    report_dir = os.path.join(green_machine_path, f'BTCUSDT/@depth10@100ms/reports/{date}/')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Write interruption data to a file
    report_file = os.path.join(report_dir, f'downtime_report_{date}.txt')
    with open(report_file, 'w') as f:
        for interruption in interruption_log:
            start = interruption['start'].strftime('%H:%M:%S')
            end = interruption['end'].strftime('%H:%M:%S') if interruption['end'] else 'Ongoing'
            duration = (interruption['end'] - interruption['start']).total_seconds() if interruption['end'] else 'N/A'
            f.write(f"Interruption from {start} to {end}, Duration: {duration} seconds\n")

    # Generate a downtime visualization
    downtime_intervals = [(i['start'], i['end']) for i in interruption_log if i['end']]
    start_times = [start.timestamp() for start, end in downtime_intervals]
    end_times = [end.timestamp() for start, end in downtime_intervals]

    if start_times and end_times:
        plt.figure(figsize=(10, 6))
        plt.hlines(1, min(start_times), max(end_times), colors='green', label='Uptime')
        plt.hlines(1, start_times, end_times, colors='red', lw=6, label='Downtime')
        plt.xlabel('Timestamp')
        plt.ylabel('Status')
        plt.title('WebSocket Connection Downtime')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(report_dir, f'downtime_visualization_{date}.png'))
        plt.close()

    print(f"Report and downtime visualization generated at: {report_dir}")

def start_websocket():
    # Set the WebSocket endpoint
    socket = 'wss://stream.binance.com:9443/ws'
    
    # Initialize the WebSocketApp
    ws = websocket.WebSocketApp(socket,
                                on_open=on_open,
                                on_message=on_message,
                                on_close=on_close,
                                on_error=on_error)
    
    # Run the WebSocketApp
    try:
        ws.run_forever()
    except Exception as e:
        print(f"WebSocket run_forever error: {e}")
        restart_script()

if __name__ == "__main__":
    try:
        start_websocket()
    finally:
        end_interruption()
        generate_report()  # Generate report at the end of the day