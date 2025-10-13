import json
import pandas as pd
import os
from natsort import natsorted
from datetime import datetime
from glob import glob

# Define base directories for raw data and output data
BASE_DIR_RAW = '/home/gaen/Documents/codespace-gaen/Simons/raw'
BASE_DIR_OUTPUT = '/home/gaen/Documents/codespace-gaen/Simons/input_data'

def get_aggtrade_dict(filepath):
    aggtrade = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.replace("'", '"')
            record = json.loads(line)
            if 'aggTrade' in line:
                # Add datetime conversion if necessary
                record['datetime'] = datetime.fromtimestamp(float(record['datetime']))
                aggtrade.append(record)
    return aggtrade

def expand_aggtrade(aggtrade):
    aggtrade_pd = pd.DataFrame(aggtrade)
    return aggtrade_pd

def run(filepath):
    aggtrade = get_aggtrade_dict(filepath)
    aggtrade_expanded = expand_aggtrade(aggtrade)
    return aggtrade_expanded

dates = ['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024']
dates = ['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024','30-Aug-2024','31-Aug-2024','01-Sep-2024','02-Sep-2024','03-Sep-2024','04-Sep-2024','05-Sep-2024','06-Sep-2024','07-Sep-2024']


for date in dates:
    output_dir = os.path.join(BASE_DIR_OUTPUT, date)
    if os.path.isfile(os.path.join(output_dir, 'aggTrade.csv')):
        print(f'Skipping files from {date} as they already exist.\n')
        continue

    print(f'Processing files from {date}.\n')

    input_path = os.path.join(BASE_DIR_RAW, date, 'orderbook*')
    files_to_process = natsorted(glob(input_path))
    print(files_to_process)

    os.makedirs(output_dir, exist_ok=True)
    
    aggtrade = run(files_to_process[0])
  
    if len(files_to_process) > 1:
        for file in files_to_process[1:]:
            aggtrade = pd.concat([aggtrade, run(file)], axis=0)
  
    # Convert 'datetime' column to datetime64 and sort the DataFrame by it
    aggtrade['datetime'] = pd.to_datetime(aggtrade['datetime'])
    sorted_aggtrade = aggtrade.sort_values(by='datetime')

    # Save the sorted DataFrame to a CSV file
    sorted_aggtrade.to_csv(os.path.join(output_dir, 'aggTrade.csv'), index=False)