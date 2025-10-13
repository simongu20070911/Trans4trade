import pandas as pd
import os
import time
from datetime import timedelta

BASE_DIR = '/home/gaen/Documents/codespace-gaen/Simons/input_data'
DATES_TO_PROCESS = ['23-Aug-2024', '24-Aug-2024', '25-Aug-2024', '26-Aug-2024', '27-Aug-2024', '29-Aug-2024']

def load_and_preprocess_data(file_path, timestamp_column, convert_to_ms=False):
    df = pd.read_csv(file_path)
    
    # Handle timestamp conversion with full precision
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit='ms' if convert_to_ms else 's', exact=True)
    
    return df.sort_values(timestamp_column)

def process_date(date_str):
    dir_path = os.path.join(BASE_DIR, date_str)
    orderbook_file = os.path.join(dir_path, 'orderbook.csv')
    aggtrade_file = os.path.join(dir_path, 'aggTrade.csv')
    output_file = os.path.join(dir_path, 'orderbook_agg_trade.csv')
    
    print(f"Processing date: {date_str}")

    if os.path.exists(output_file):
        print(f"Skipped {date_str}: {output_file} already exists.")
        return

    if not all(os.path.exists(f) for f in [orderbook_file, aggtrade_file]):
        print(f"Skipped {date_str}: Missing input file(s).")
        return

    print(f"Loading and preprocessing data for {date_str}...")
    orderbook_df = load_and_preprocess_data(orderbook_file, 'datetime', convert_to_ms=False)
    aggtrade_df = load_and_preprocess_data(aggtrade_file, 'E', convert_to_ms=True)

    print("Orderbook sample:")
    print(orderbook_df['datetime'].head())
    print("\nAggTrade sample (after adjustment):")
    print(aggtrade_df['E'].head())

    orderbook_df['index_id'] = range(1, len(orderbook_df) + 1)

    total_entries = len(aggtrade_df)
    print(f"Matching and combining data for {date_str}... Total entries: {total_entries}")

    start_time = time.time()
    combined_data = []

    orderbook_idx = 0
    orderbook_len = len(orderbook_df)

    for i, trade_row in aggtrade_df.iterrows():
        event_time = trade_row['E']
        
        # Collect all relevant orderbook rows that match or are closest to the trade event
        matching_orderbooks = []
        
        while orderbook_idx < orderbook_len and orderbook_df.iloc[orderbook_idx]['datetime'] <= event_time:
            if orderbook_df.iloc[orderbook_idx]['datetime'] == event_time:
                matching_orderbooks.append(orderbook_df.iloc[orderbook_idx])
            orderbook_idx += 1

        # Handle the case where no exact match was found
        if not matching_orderbooks and orderbook_idx > 0:
            matching_orderbooks.append(orderbook_df.iloc[orderbook_idx - 1])

        for closest_orderbook_row in matching_orderbooks:
            combined_row = {
                'datetime_x': closest_orderbook_row['datetime'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'lastUpdateId': closest_orderbook_row['lastUpdateId'],
                'match_id': closest_orderbook_row['index_id'],
                **{col: closest_orderbook_row[col] for col in closest_orderbook_row.index if col.startswith(('ask', 'bid'))},
                'eventtype': trade_row['e'],
                'eventtime': trade_row['E'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'symbol': trade_row['s'],
                'sellID': trade_row['a'],
                'price': trade_row['p'],
                'quantity': trade_row['q'],
                'firsttradeID': trade_row['f'],
                'lasttradeID': trade_row['l'],
                'tradetime': trade_row['T'],
                'marketmaker': trade_row['m'],
                'ignore': trade_row['M'],
                'datetime_y': trade_row['E'].strftime('%Y-%m-%d %H:%M:%S.%f')
            }

            combined_data.append(combined_row)

        if i % 100 == 0:  # Update progress every 100 entries
            percent_complete = (i / total_entries) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_entries / (i + 1)
            time_remaining = max(estimated_total_time - elapsed_time, 0)
            time_remaining_str = str(timedelta(seconds=int(time_remaining)))
            print(f"Processing {date_str}: {percent_complete:.2f}% complete, Time remaining: {time_remaining_str}", end='\r')

    combined_df = pd.DataFrame(combined_data)

    print(f"\nSaving output to {output_file} for {date_str}...")
    combined_df.to_csv(output_file, index=False)
    print(f"Processed {date_str}: Output saved to {output_file}")

def main():
    for date_str in DATES_TO_PROCESS:
        process_date(date_str)
    print("Processing complete.")

if __name__ == "__main__":
    main()