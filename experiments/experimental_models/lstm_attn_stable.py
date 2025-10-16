import os
import glob

from tqdm import trange
import random
import math

from dateutil import parser
from datetime import datetime
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#####################################
#        HYPERPARAMETERS            #
#####################################

# Feature Configuration
FEATURES = [
    'price', 'lag_return',
    'bid1', 'bidqty1', 'bid2', 'bidqty2', 'bid3', 'bidqty3', 'bid4', 'bidqty4', 'bid5', 'bidqty5',
    'bid6', 'bidqty6', 'bid7', 'bidqty7', 'bid8', 'bidqty8', 'bid9', 'bidqty9',
    'ask1', 'askqty1', 'ask2', 'askqty2', 'ask3', 'askqty3', 'ask4', 'askqty4', 'ask5', 'askqty5',
    'ask6', 'askqty6', 'ask7', 'askqty7', 'ask8', 'askqty8', 'ask9', 'askqty9'
]

# Data Processing Parameters
WINDOW_SIZE = 100          # Number of time steps to look back. Larger values capture longer patterns but increase memory usage
LAG = 1                    # Prediction horizon. How many steps ahead to predict
TRAIN_SPLIT = 0.80        # Ratio of data used for training vs validation

# Model Architecture Parameters
INPUT_SIZE = len(FEATURES)  # Number of input features (automatically set based on FEATURES list)
HIDDEN_SIZE = 32          # Size of LSTM hidden states. Larger values increase model capacity but risk overfitting
NUM_LSTM_LAYERS = 5       # Number of stacked LSTM layers. More layers can capture more complex patterns
DROPOUT_RATE = 0.25       # Dropout probability. Higher values (0.2-0.5) help prevent overfitting
OUTPUT_SIZE = 1           # Number of output values to predict

# Attention Parameters
ATTENTION_SIZE = HIDDEN_SIZE  # Size of attention layer. Usually same as hidden size
ATTENTION_DROPOUT = 0.1       # Dropout specific to attention layer

# Training Parameters
BATCH_SIZE = 1024         # Batch size for training. Larger batches give more stable gradients but use more memory
NUM_EPOCHS = 50           # Number of training epochs. Increase if model is underfitting
LEARNING_RATE = 0.005     # Initial learning rate. Too high -> unstable, Too low -> slow convergence
SCHEDULER_STEP = 10       # Number of epochs before learning rate adjustment
SCHEDULER_GAMMA = 0.995   # Factor to multiply learning rate at each step. <1 for decay

# Model Saving Parameters
current_working_directory = '/home/gaen/Documents/codespace-gaen/Ts-master'
current_date = datetime.now().date()
model_add_path = f'playround_models/lstm_attention/{current_date}'

if not os.path.exists(current_working_directory + '/' + model_add_path):
    os.makedirs(current_working_directory + '/' + model_add_path)

# Configuration Dictionary
config = {
    "plots": {
        "show_plots": False,
        "xticks_interval": 1200,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_test": "#FF4136",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136"
    },
    "data": {
        "window_size": WINDOW_SIZE,
        "train_split_size": TRAIN_SPLIT,
    },
    "model_MO": {
        "input_size": INPUT_SIZE,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "lstm_size": HIDDEN_SIZE,
        "dropout": DROPOUT_RATE,
        "output_size": OUTPUT_SIZE,
        "attention_size": ATTENTION_SIZE,
        "attention_dropout": ATTENTION_DROPOUT
    },
    "training_MO": {
        "batch_size": BATCH_SIZE,
        "num_epoch": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "scheduler_step_size": SCHEDULER_STEP,
        "scheduler_gamma": SCHEDULER_GAMMA
    },
    "paths": {
        "drive": {
            "agg_trade": {
                "train": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
                "test": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/"
            },
            "orderbook": {
                "train": "/content/drive/MyDrive/IP/Repos/LSTM_Transformer/input_data/orderbook_clean.csv",
                "test": "/content/drive/MyDrive/IP/Repos/LSTM_Transformer/input_data/orderbook_test_clean.csv"
            },
            "models": "/content/drive/MyDrive/IP/Repos/LSTM_Transformer/models/",
            "figures": "/content/drive/MyDrive/IP/Repos/LSTM_Transformer/figures/"
        },
        "local": {
            "agg_trade": {
                "train": "./data/input_data/",
                "test": "./data/input_data/"
            },
            "orderbook": {
                "train": "./data/input_data/orderbook_clean.csv",
                "test": "./data/input_data/orderbook_test_clean.csv"
            },
            "models": "./models/",
            "figures": "./figures"
        }
    }
}

#####################################
#      PARAMETER TUNING GUIDE       #
#####################################
'''
How to tune the hyperparameters:

1. Data Window (WINDOW_SIZE):
   - Start with size that makes sense for your time series (e.g., daily data -> 5-20 days)
   - Increase if model isn't capturing long-term patterns
   - Decrease if training is too slow or memory intensive

2. Model Capacity (HIDDEN_SIZE, NUM_LSTM_LAYERS):
   - Start with HIDDEN_SIZE = 32 or 64
   - Start with NUM_LSTM_LAYERS = 2
   - If underfitting (high train & val loss): increase either/both
   - If overfitting (low train loss, high val loss): decrease either/both

3. Regularization (DROPOUT_RATE, ATTENTION_DROPOUT):
   - Start with 0.2-0.3
   - If overfitting: increase dropouts (max 0.5)
   - If underfitting: decrease dropouts

4. Training Parameters:
   - BATCH_SIZE: Start with 256-1024, increase if training is unstable
   - LEARNING_RATE: Start with 0.001, adjust if:
     * Loss exploding -> decrease
     * Learning too slow -> increase
   - NUM_EPOCHS: Train until validation loss plateaus
   - SCHEDULER_STEP/GAMMA: Adjust if learning rate decay is too fast/slow

5. Attention Parameters:
   - ATTENTION_SIZE: Usually keep same as HIDDEN_SIZE
   - Can be decreased to reduce memory usage

Monitor:
- Training loss vs Validation loss
- R2 score on validation set
- Learning rate changes over time
- Attention weights distribution
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

### load data

date_train = 'All_to_Sept' 
date_test = 'All_to_Sept'
drive = None
if drive:
    agg_trade = pd.read_csv(config["paths"]["drive"]["agg_trade"]["train"]+date_train+'/orderbook.csv')    
else:
    agg_trade = pd.read_csv(config["paths"]["local"]["agg_trade"]["train"]+date_train+'/orderbook_agg_trade_dollarvol.csv')
    # agg_trade_test = pd.read_csv(config["paths"]["local"]["agg_trade"]["test"]+date_test+'/orderbook_agg_trade_dollarvol.csv')

#agg_trade['price'] = agg_trade['w_midprice']

###prepare data for MO LSTM

def prepare_data_x(data, window_size, lag):
    '''
    Windows the input data for the ML models.
    '''
    n_row = data.shape[0] - window_size + 1
    subset = data[:window_size]
    subset_mean = np.mean(subset, axis=0)
    output = np.zeros([n_row, window_size, len(subset_mean)])
    x_mean = np.zeros([n_row, len(subset_mean)])
    x_std = np.zeros([n_row, len(subset_mean)])
    for idx in range(n_row):
        subset = data[idx:idx+window_size]
        subset_mean = np.mean(subset, axis=0)
        subset_std = np.std(subset, axis=0) + 0.01
        subset_norm = (subset-subset_mean)/subset_std
        x_mean[idx,:] = subset_mean
        x_std[idx,:] = subset_std
        output[idx,:,:] = subset_norm
    x_mean = np.array(x_mean)
    x_std = np.array(x_std)
    return output[:-lag-1], output[-1], x_mean, x_std

def prepare_data_y(x, window_size, lag):
    '''
    Windows the target data for the ML models.
    '''
    output = np.zeros([len(x)-window_size-lag])
    std = 1.1*np.sqrt(lag)+lag*0.01
    for idx in range(0,len(x)-window_size-lag):
        output[idx] = np.log(x[window_size+lag-1+idx,0]/x[window_size-1+idx,0])*10_000
    output = output/std
    return output

def prepare_data(normalized_prices_train, dates_train, normalized_prices_test, dates_test, config, lag=1, plot=False):
    '''
    Returns input and target data.
    '''
    data_x, data_x_unseen, x_mean, x_std = prepare_data_x(normalized_prices_train, window_size=config["data"]["window_size"], lag=lag)
    data_y = prepare_data_y(normalized_prices_train, window_size=config["data"]["window_size"], lag=lag)

    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val


#defining the Many-to-one LSTM
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class LSTM_MO(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_layer_size=HIDDEN_SIZE, 
                 num_layers=NUM_LSTM_LAYERS, output_size=OUTPUT_SIZE, 
                 dropout=DROPOUT_RATE, attention_dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Input transformation
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.prelu = nn.PReLU()
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=hidden_layer_size, 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        
        # Attention layers
        self.attention_weights = nn.Linear(hidden_layer_size, 1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_weights.weight)

    def attention(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention scores
        attention_weights = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initial transformation
        x = self.linear_1(x)
        x = self.prelu(x)
        
        # LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Apply dropout and final linear layer
        x = self.dropout(context_vector)
        predictions = self.linear_2(x)
        
        return predictions.squeeze(-1)  # Ensure output is [batch_size]

    def train_model(self, train_dataloader, val_dataloader, learning_rate, scheduler_step_size, n_epochs=50, device="cpu", save_path=None, forecast_window=None):
      
        def run_epoch(dataloader, is_training=False):
            epoch_loss = 0
            outputs = torch.Tensor(0).to(device)
            targets = torch.Tensor(0).to(device)
            if is_training:
                self.train()
            else:
                self.eval()
            for idx, (x, y) in enumerate(dataloader):
                if is_training:
                    optimizer.zero_grad()

                batchsize = x.shape[0]
                x = x.to(device)
                y = y.to(device)
                out = self.forward(x)
                loss = criterion(out.contiguous(), y.contiguous())

                if is_training:
                    loss.backward()
                    optimizer.step()
                
                if not is_training:
                    outputs = torch.cat((outputs.contiguous(), out.detach()))
                    targets = torch.cat((targets, y.contiguous()))
                    
                epoch_loss += (loss.detach().item() / batchsize)
                
            lr = scheduler.get_last_lr()[0]
            if not is_training:
                print(outputs.cpu().detach().numpy())
                print(targets.cpu().detach().numpy())
                plt.plot(targets.cpu().detach().numpy(), alpha=0.3)
                plt.plot(outputs.cpu().detach().numpy())
                plt.show()
                r2 = r2_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                return epoch_loss, lr, r2
            else:
                return epoch_loss, lr

      
        # define optimizer, scheduler and loss function
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.AdamW(model_MO.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.995)

        # begin training
        for epoch in range(n_epochs):
            loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
            loss_val, lr_val, r2 = run_epoch(val_dataloader)
            scheduler.step()

            if save_path:
                results = {
                        'model': 'LSTM_MO',
                        'pred_len': forecast_window,
                        'epoch': epoch,
                        'train_loss': loss_train,
                        'val_loss': loss_val,
                        'r2_val_sklearn': r2            
                }

                df = pd.DataFrame([results])
                df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

            
            print('Epoch[{}/{}] | loss train:{:.6f}, val loss:{:.6f} | lr:{:.6f} | r2: {:.5f}|'
                      .format(epoch+1, n_epochs, loss_train, loss_val, lr_train, r2))





##model training:

def augment_trade_data(df, lag=0, forecast_window=None):
    if forecast_window:
        df['lag_return'] = np.log(df['price'].shift(forecast_window)/df['price'].shift(forecast_window+1))
        return df.iloc[forecast_window+1:,:]
    if lag == 0:
        return df
    else:
        col_name = 'log_lag'+str(lag)+'_price'
        df[col_name] = np.log(df.price) - np.log(df.price).shift(lag)
        return df.iloc[lag:,:]


# lag=1
save_path = os.path.join(f'{current_working_directory}/{model_add_path}',
                            str(int(time.time()))+'_results.csv')
# Create the directory if it doesn't exist

forecast_windows = [i for i in range(1,32)]


for lag in forecast_windows:
    orderbook = augment_trade_data(agg_trade, forecast_window=lag)
    
    # Use the predefined FEATURES list
    split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
        np.array(orderbook[FEATURES][0:100000]),
        np.array(agg_trade.datetime[2_005_000:2_006_000]),
        np.array(orderbook[FEATURES][60_000:70_000]),
        np.array(agg_trade.datetime[60_000:70_000]),
        config,
        lag=lag
    )

    # Create datasets using the config parameters
    train_dataset = TimeSeriesDataset(data_x_train, data_y_train)
    val_dataset = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=config["training_MO"]["batch_size"],
                                shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                               batch_size=config["training_MO"]["batch_size"],
                               shuffle=False)

    # Use the config parameters for model initialization
    model_MO = LSTM_MO(
        input_size=config["model_MO"]["input_size"],
        hidden_layer_size=config["model_MO"]["lstm_size"],
        num_layers=config["model_MO"]["num_lstm_layers"],
        output_size=config["model_MO"]["output_size"],
        dropout=config["model_MO"]["dropout"],
        attention_dropout=config["model_MO"]["attention_dropout"]
    )
    
    model_MO = model_MO.to(device)

    model_MO.train_model(train_dataloader=train_dataloader, val_dataloader=val_dataloader, learning_rate=0.001,
                scheduler_step_size=config["training_MO"]["scheduler_step_size"], n_epochs=5,
                device='cuda', save_path=save_path, forecast_window=lag)

    # date_now = datetime.now()
    # timestamp = date_now.strftime("%d-%b-%Y_%H:%M:%S.%f")
    del data_x_train 
    del data_y_train
    del data_x_val
    del data_y_val

    # torch.save(model_MO, f'{current_working_directory}/othermodels/{model_add_path}/No.1/LSTM_MO_LAG_{lag}.pt')
    model_save_path = f'{current_working_directory}/{model_add_path}/No.2'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model_MO, f'{model_save_path}/LSTM_MO_LAG_{lag}.pt')
    print(f'Done with prediction len {lag}.')