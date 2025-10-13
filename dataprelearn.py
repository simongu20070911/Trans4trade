
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import cleaner
import utils.plotter as plotter
dates = ['09-Jun-2022','10-Jun-2022','11-Jun-2022','12-Jun-2022','13-Jun-2022','14-Jun-2022','16-Jun-2022','17-Jun-2022']
dates = ['15-Aug-2024','16-Aug-2024']
dates = ['15-Aug-2024', '16-Aug-2024','19-Aug-2024','20-Aug-2024','21-Aug-2024']

dates = ['09-Jun-2022']

 
orderbook = []

for idx, date in enumerate(dates):
    orderbook.append(pd.read_csv(f'./input_data/{date}/orderbook.csv'))

for idx, date in enumerate(dates):
    orderbook[idx]['w_midprice'] = (orderbook[idx]['ask1']*orderbook[idx]['askqty1']+orderbook[idx]['bid1']*orderbook[idx]['bidqty1'])/(orderbook[idx]['askqty1']+orderbook[idx]['bidqty1'])

orderbook_all = pd.concat(orderbook)

orderbook_all.to_csv('./input_data/all/orderbook.csv')

orderbook_all.set_index('datetime', inplace=True)
orderbook_all.sort_index(ascending=True, inplace=True)
plt.plot(orderbook_all.w_midprice.to_list())
from IPython.display import clear_output
import time
li = orderbook_all.w_midprice.to_list()
print(orderbook_all.shape)
#for a in range(10000):
#    plt.plot(li[a:a+1000])
#    #wait for 1 sec
a =3000
plt.plot(li[a:a+160000])

