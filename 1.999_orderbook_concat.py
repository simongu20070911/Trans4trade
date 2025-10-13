import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import cleaner
dates =['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024']



orderbook = []

for idx, date in enumerate(dates):
    orderbook.append(pd.read_csv(f'./input_data/{date}/orderbook.csv'))

for idx, date in enumerate(dates):
    orderbook[idx]['price'] = (orderbook[idx]['ask1']*orderbook[idx]['askqty1']+orderbook[idx]['bid1']*orderbook[idx]['bidqty1'])/(orderbook[idx]['askqty1']+orderbook[idx]['bidqty1'])

orderbook_all = pd.concat(orderbook)
orderbook_all.index = orderbook_all['datetime']

orderbook_all.shape
orderbook_all.to_csv('./input_data/All_to_Sept/orderbook.csv')