import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import cleaner

import utils.plotter as plotter


dates = ['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024','30-Aug-2024','31-Aug-2024','01-Sep-2024','02-Sep-2024','03-Sep-2024','04-Sep-2024','05-Sep-2024','06-Sep-2024','07-Sep-2024']
folder_name = "All_to_Sept"
agg_trade_orderbook = []

for date in dates:
  agg_trade_orderbook.append(pd.read_csv(f'./input_data/{date}/orderbook_agg_trade.csv'))

fig, ax = plt.subplots(3, 3, figsize = (20, 20))

plot_idx = 0
for idx in range(3):
  for jdx in range(3):
    rand = np.random.randint(0, len(agg_trade_orderbook[plot_idx])-5000, size=1)
    ax2 = ax[idx][jdx].twinx()
    ax[idx][jdx].plot(agg_trade_orderbook[plot_idx].price[rand[0]:rand[0]+5000])
    ax2.plot(agg_trade_orderbook[plot_idx].quantity[rand[0]:rand[0]+5000], color='orange')
    plot_idx += 1

for idx in range(len(dates)):
  mean_price = np.mean(agg_trade_orderbook[idx].price)
  std_price = np.std(agg_trade_orderbook[idx].price)
  mean_qty = np.mean(agg_trade_orderbook[idx].quantity)
  std_qty = np.std(agg_trade_orderbook[idx].quantity)
  mean_dollarvol =  np.mean(agg_trade_orderbook[idx].price*agg_trade_orderbook[idx].quantity)
  std_dollarvol =  np.std(agg_trade_orderbook[idx].price*agg_trade_orderbook[idx].quantity)
  print(f'| date: {dates[idx]} | mean price: {mean_price:0.2f} ~ {std_price:0.2f} | \
mean qty {mean_qty:0.5f} ~ {std_qty:0.5f} | mean dollar vol {mean_dollarvol:0.2f} ~ {std_dollarvol:0.2f}|')
  

agg_trade_orderbook_all = pd.concat(agg_trade_orderbook)

agg_trade_orderbook_all['dollarvol'] = agg_trade_orderbook_all.quantity * agg_trade_orderbook_all.price

bid_cols = ['bid'+str(i) for i in range(1,11)]+['bidqty'+str(i) for i in range(1,11)]
ask_cols = ['ask'+str(i) for i in range(1,11)]+['askqty'+str(i) for i in range(1,11)]

bid_ask_cols = bid_cols+ask_cols

cols_dict = {'price':'price',
             'quantity':'quantity',
             'datetime':'datetime_y',
             'bid_ask_columns':bid_ask_cols}

median_dollar_vol = np.median(agg_trade_orderbook_all['dollarvol'])
mean_dollar_vol = np.mean(agg_trade_orderbook_all['dollarvol'])

orderbook_clean_dollarvol_all = cleaner.group_book_by_dollarvol2(agg_trade_orderbook_all, cols_dict, mean_dollar_vol)


orderbook_clean_dollarvol_all.to_csv(f'./input_data/{folder_name}/orderbook_agg_trade_dollarvol.csv', index=False)