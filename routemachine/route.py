import sys
import os

# Get the absolute path to the directory containing 'the-green-machine'
green_machine_path = os.path.abspath("/home/gaen/Documents/codespace-gaen/the-green-machine")
get_order_book_dict_path = os.path.abspath('/home/gaen/Documents/codespace-gaen/the-green-machine/system/BTCUSDT/@depth10@100ms')
# Append the path to 'sys.path'


def get_order_book_dict(): 
    sys.path.append(get_order_book_dict_path)
def get_workplace_path():
    sys.path.append(green_machine_path)
