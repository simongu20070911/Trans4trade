import numpy as np
import json
import pandas as pd
import os
from natsort import natsorted
from datetime import datetime
from glob import glob

# Define base directories for raw data and output data
BASE_DIR_RAW = '/home/gaen/Documents/codespace-gaen/the-green-machine/system/BTCUSDT/@depth10@100ms/raw'
BASE_DIR_RAW = '/home/gaen/Documents/codespace-gaen/Simons/raw'

BASE_DIR_OUTPUT = '/home/gaen/Documents/codespace-gaen/Simons/input_data'

def get_orderbook_dict(filepath):
    orderbook = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.replace("'", '"')
            record = json.loads(line)
            if 'lastUpdateId' in line:
                original_datetime = float(record['datetime'])  # Store as a float (Unix timestamp)
                record['datetime'] = original_datetime
                record['expanded_datetime'] = datetime.utcfromtimestamp(original_datetime)
                orderbook[original_datetime] = record
    return orderbook

def expand_orderbook(orderbook, levels=10):
    datetime_arr = np.zeros([len(orderbook)], dtype='float64')  # Use float64 to store original datetime
    expanded_datetime_arr = np.zeros([len(orderbook)], dtype='datetime64[ms]')
    lastUpdateId = np.zeros([len(orderbook)], dtype='int64')
    bids = np.zeros([len(orderbook), 20])
    asks = np.zeros([len(orderbook), 20])

    record_idx = 0
    for key, value in orderbook.items():
        datetime_arr[record_idx] = key
        expanded_datetime_arr[record_idx] = np.datetime64(value['expanded_datetime'])
        lastUpdateId[record_idx] = value['lastUpdateId']
        bids_to_extract = value['bids']
        for idx, bid in enumerate(bids_to_extract):
            bids[record_idx, 2 * idx], bids[record_idx, 2 * idx + 1] = float(bid[0]), float(bid[1])
        asks_to_extract = value['asks']
        for idx, ask in enumerate(asks_to_extract):
            asks[record_idx, 2 * idx], asks[record_idx, 2 * idx + 1] = float(ask[0]), float(ask[1])
        record_idx += 1

    orderbook_dict = {'datetime': datetime_arr,
                      'expanded_datetime': expanded_datetime_arr,
                      'lastUpdateId': lastUpdateId}

    for level in range(levels):
        orderbook_dict['ask' + str(level + 1)] = asks[:, 2 * level]
        orderbook_dict['askqty' + str(level + 1)] = asks[:, 2 * level + 1]

    for level in range(levels):
        orderbook_dict['bid' + str(level + 1)] = bids[:, 2 * level]
        orderbook_dict['bidqty' + str(level + 1)] = bids[:, 2 * level + 1]

    orderbook_pd = pd.DataFrame(orderbook_dict)
    return orderbook_pd

def run(filepath):
    orderbook = get_orderbook_dict(filepath)
    orderbook_expanded = expand_orderbook(orderbook)
    return orderbook_expanded

dates = ['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','28-Aug-2024','29-Aug-2024']
dates = ['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024','30-Aug-2024','31-Aug-2024','01-Sep-2024','02-Sep-2024','03-Sep-2024','04-Sep-2024','05-Sep-2024','06-Sep-2024','07-Sep-2024']
'''dates = [
    '09-Jun-2022', 
    '10-Jun-2022', 
    '11-Jun-2022', 
    '12-Jun-2022', 
    '13-Jun-2022', 
    '14-Jun-2022', 
    '16-Jun-2022', 
    '17-Jun-2022', 
    '18-Jun-2022', 
    '19-Jun-2022',
]'''
#dates = ['15-Aug-2024','16-Aug-2024']

for date in dates:
    output_dir = os.path.join(BASE_DIR_OUTPUT, date)
    if os.path.isfile(os.path.join(output_dir, 'orderbook.csv')):
        print(f'Skipping files from {date} as they already exist.\n')
        continue

    print(f'Processing files from {date}.\n')

    input_path = os.path.join(BASE_DIR_RAW, date, 'orderbook*')
    files_to_process = natsorted(glob(input_path))
    print(files_to_process)

    os.makedirs(output_dir, exist_ok=True)
    
    orderbook = run(files_to_process[0])
  
    if len(files_to_process) > 1:
        for file in files_to_process[1:]:
            orderbook = pd.concat([orderbook, run(file)], axis=0)
  
    # Sort the DataFrame by the 'datetime' column (original Unix timestamp)
    sorted_df = orderbook.sort_values(by='datetime')

    # Save the sorted DataFrame to a CSV file
    sorted_df.to_csv(os.path.join(output_dir, 'orderbook.csv'), index=False)