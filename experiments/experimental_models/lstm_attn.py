"""
LSTM with Attention for Time Series Prediction
This module implements an LSTM model with attention mechanism for time series forecasting.

Key Components:
- LSTM layers for sequence processing
- Attention mechanism for focusing on relevant parts of the sequence
- Configurable hyperparameters for easy tuning
"""

import os
import glob
from datetime import datetime
import time
import random
import math

# Data processing imports
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import trange

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Metrics and visualization
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#####################################
#        CONFIGURATIONS             #
#####################################

class Config:
    """Configuration class to organize all parameters"""
    
    # System paths
    BASE_DIR = '/home/gaen/Documents/codespace-gaen/Ts-master'
    CURRENT_DATE = datetime.now().date()
    MODEL_SAVE_PATH = f'playround_models/lstm_attention/{CURRENT_DATE}'
    
    # Data Source Configuration
    DATA_SOURCE = {
        "USE_DRIVE": False,  # If True, use drive paths, if False use local paths
        #"TRAIN_DATE": "4000longshot",
        "TRAIN_DATE": "All_to_Sept",
        "FILE_NAME": "orderbook_agg_trade_dollarvol.csv",
        "DRIVE_FILE_NAME": "orderbook.csv"  # File name when using drive
    }
    
    # Feature Configuration
    FEATURES = [
        'price', 'lag_return',
        'bid1', 'bidqty1', 'bid2', 'bidqty2', 'bid3', 'bidqty3', 'bid4', 'bidqty4', 'bid5', 'bidqty5',
        'bid6', 'bidqty6', 'bid7', 'bidqty7', 'bid8', 'bidqty8', 'bid9', 'bidqty9',
        'ask1', 'askqty1', 'ask2', 'askqty2', 'ask3', 'askqty3', 'ask4', 'askqty4', 'ask5', 'askqty5',
        'ask6', 'askqty6', 'ask7', 'askqty7', 'ask8', 'askqty8', 'ask9', 'askqty9'
    ]

    # Data Processing Parameters
    TRAIN_SIZE = 400000    # Training data size
    VAL_SIZE = 10000      # Validation data size
    WINDOW_SIZE = 100     # Number of time steps to look back
    LAG = 1              # Prediction horizon
    TRAIN_SPLIT = 0.80   # Train/validation split ratio
    FORECAST_WINDOWS = list(range(1, 32))  # Forecast windows
    
    # Model Architecture
    INPUT_SIZE = len(FEATURES)
    HIDDEN_SIZE = 32
    NUM_LSTM_LAYERS = 5
    DROPOUT_RATE = 0.25
    OUTPUT_SIZE = 1
    
    # Attention Parameters
    ATTENTION_SIZE = HIDDEN_SIZE
    ATTENTION_DROPOUT = 0.1
    
    # Training Parameters
    BATCH_SIZE = 2048
    NUM_EPOCHS = 60
    LEARNING_RATE = 0.001
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.995
    
    # Visualization Parameters
    PLOT_CONFIG = {
        "show_plots": False,
        "xticks_interval": 1200,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_test": "#FF4136",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136"
    }
    
    # File Paths
    PATHS = {
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

    @classmethod
    def create_model_save_dir(cls):
        """Create directory for saving models if it doesn't exist"""
        save_path = os.path.join(cls.BASE_DIR, cls.MODEL_SAVE_PATH)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    @classmethod
    def get_config_dict(cls):
        """Convert configuration to dictionary format"""
        return {
            "data": {
                "window_size": cls.WINDOW_SIZE,
                "train_split_size": cls.TRAIN_SPLIT,
                "train_size": cls.TRAIN_SIZE,
                "val_size": cls.VAL_SIZE,
                "forecast_windows": cls.FORECAST_WINDOWS
            },
            "model_MO": {
                "input_size": cls.INPUT_SIZE,
                "num_lstm_layers": cls.NUM_LSTM_LAYERS,
                "lstm_size": cls.HIDDEN_SIZE,
                "dropout": cls.DROPOUT_RATE,
                "output_size": cls.OUTPUT_SIZE,
                "attention_size": cls.ATTENTION_SIZE,
                "attention_dropout": cls.ATTENTION_DROPOUT
            },
            "training_MO": {
                "batch_size": cls.BATCH_SIZE,
                "num_epoch": cls.NUM_EPOCHS,
                "learning_rate": cls.LEARNING_RATE,
                "scheduler_step_size": cls.SCHEDULER_STEP,
                "scheduler_gamma": cls.SCHEDULER_GAMMA
            },
            "plots": cls.PLOT_CONFIG,
            "paths": cls.PATHS
        }

    @classmethod
    def get_data_path(cls):
        """Get the appropriate data path based on configuration"""
        if cls.DATA_SOURCE["USE_DRIVE"]:
            return os.path.join(
                cls.PATHS["drive"]["agg_trade"]["train"],
                cls.DATA_SOURCE["TRAIN_DATE"],
                cls.DATA_SOURCE["DRIVE_FILE_NAME"]
            )
        else:
            return os.path.join(
                cls.PATHS["local"]["agg_trade"]["train"],
                cls.DATA_SOURCE["TRAIN_DATE"],
                cls.DATA_SOURCE["FILE_NAME"]
            )

#####################################
#           DATA CLASSES            #
#####################################

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data"""
    
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

#####################################
#           MODEL CLASS             #
#####################################

class LSTM_MO(nn.Module):
    """LSTM model with Transformer encoder and attention mechanism for time series prediction"""
    
    def __init__(self, input_size=Config.INPUT_SIZE, 
                 hidden_layer_size=Config.HIDDEN_SIZE,
                 num_layers=Config.NUM_LSTM_LAYERS, 
                 output_size=Config.OUTPUT_SIZE,
                 dropout=Config.DROPOUT_RATE,
                 nhead=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        # Input transformation
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.prelu = nn.PReLU()
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_layer_size,
            nhead=nhead,
            dim_feedforward=hidden_layer_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_layer_size, 
                           hidden_size=hidden_layer_size,
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_layer_size)
        self.layer_norm2 = nn.LayerNorm(hidden_layer_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        # Input transformation
        x = self.linear_1(x)  # [batch_size, seq_len, input_size] -> [batch_size, seq_len, hidden_size]
        x = self.prelu(x)
        
        # Transformer encoder processing
        x = self.layer_norm1(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size] -> [seq_len, batch_size, hidden_size]
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size] -> [batch_size, seq_len, hidden_size]
        transformer_out = transformer_out + x.permute(1, 0, 2)  # Residual connection
        
        # LSTM processing
        lstm_out, _ = self.lstm(transformer_out)  # [batch_size, seq_len, hidden_size]
        
        # Apply layer normalization
        lstm_out = self.layer_norm2(lstm_out)
        
        # Get the last time step
        last_step = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Output transformation
        out = self.dropout(last_step)
        out = self.linear_2(out)
        
        return out

    def train_model(self, train_dataloader, val_dataloader, learning_rate, scheduler_step_size, n_epochs=50, device="cpu", save_path=None, forecast_window=None):
        def run_epoch(dataloader, is_training=False):
            epoch_loss = 0
            outputs = []
            targets = []
            if is_training:
                self.train()
            else:
                self.eval()
            for idx, (x, y) in enumerate(dataloader):
                if is_training:
                    optimizer.zero_grad()

                batchsize = x.shape[0]
                x = x.to(device)
                y = y.to(device).reshape(-1, 1)  # Reshape target to match output shape

                out = self(x)
                loss = criterion(out, y)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                outputs.append(out.detach().cpu())
                targets.append(y.detach().cpu())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            r2 = r2_score(targets.numpy(), outputs.numpy())
            return epoch_loss / len(dataloader), outputs, targets, r2

        # define optimizer, scheduler and loss function
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.995)

        # begin training
        for epoch in range(n_epochs):
            loss_train, outputs_train, targets_train, r2_train = run_epoch(train_dataloader, is_training=True)
            loss_val, outputs_val, targets_val, r2_val = run_epoch(val_dataloader)
            scheduler.step()

            if save_path:
                results = {
                        'model': 'LSTM_MO',
                        'pred_len': forecast_window,
                        'epoch': epoch,
                        'train_loss': loss_train,
                        'val_loss': loss_val,
                        'r2_val_sklearn': r2_val            
                }

                df = pd.DataFrame([results])
                df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

            
            print('Epoch[{}/{}] | loss train:{:.6f}, val loss:{:.6f} | lr:{:.6f} | r2: {:.5f}|'
                      .format(epoch+1, n_epochs, loss_train, loss_val, scheduler.get_last_lr()[0], r2_val))





#####################################
#           DATA PREPARATION        #
#####################################

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


#####################################
#           DATA LOADING            #
#####################################

def preprocess_data(df, forecast_window):
    """Preprocess data by adding lag returns and other features"""
    df = df.copy()
    # Calculate lag return
    df['lag_return'] = np.log(df['price'].shift(forecast_window)/df['price'].shift(forecast_window+1))
    # Remove rows with NaN values from the lag calculation
    df = df.iloc[forecast_window+1:,:]
    return df

def load_data():
    """Load and prepare data based on configuration"""
    agg_trade = pd.read_csv(Config.get_data_path())
    return agg_trade

#####################################
#           MODEL TRAINING          #
#####################################

# Create the directory if it doesn't exist
Config.create_model_save_dir()

device = "cuda" if torch.cuda.is_available() else "cpu"

### Load data
agg_trade = load_data()

for lag in Config.FORECAST_WINDOWS:
    # Preprocess data for current forecast window
    orderbook = preprocess_data(agg_trade, lag)
    # Use the predefined FEATURES list and sizes from config
    split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
        np.array(orderbook[Config.FEATURES][0:Config.TRAIN_SIZE]),
        np.array(agg_trade.datetime[0:Config.TRAIN_SIZE]),
        np.array(orderbook[Config.FEATURES][Config.TRAIN_SIZE:Config.TRAIN_SIZE+Config.VAL_SIZE]),
        np.array(agg_trade.datetime[Config.TRAIN_SIZE:Config.TRAIN_SIZE+Config.VAL_SIZE]),
        Config.get_config_dict(),
        lag=lag
    )

    # Create datasets using the config parameters
    train_dataset = TimeSeriesDataset(data_x_train, data_y_train)
    val_dataset = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=Config.BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                               batch_size=Config.BATCH_SIZE,
                               shuffle=False)

    # Use the config parameters for model initialization
    model_MO = LSTM_MO(
        input_size=Config.INPUT_SIZE,
        hidden_layer_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout=Config.DROPOUT_RATE,
    )
    
    model_MO = model_MO.to(device)

    model_MO.train_model(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        learning_rate=Config.LEARNING_RATE,
        scheduler_step_size=Config.SCHEDULER_STEP, 
        n_epochs=Config.NUM_EPOCHS,
        device=device, 
        save_path=os.path.join(Config.BASE_DIR, Config.MODEL_SAVE_PATH, str(int(time.time()))+'_results.csv'), 
        forecast_window=lag
    )

    # date_now = datetime.now()
    # timestamp = date_now.strftime("%d-%b-%Y_%H:%M:%S.%f")
    del data_x_train 
    del data_y_train
    del data_x_val
    del data_y_val

    # torch.save(model_MO, f'{Config.BASE_DIR}/othermodels/{Config.MODEL_SAVE_PATH}/No.1/LSTM_MO_LAG_{lag}.pt')
    model_save_path = f'{Config.BASE_DIR}/{Config.MODEL_SAVE_PATH}/No.2'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model_MO, f'{model_save_path}/LSTM_MO_LAG_{lag}.pt')
    print(f'Done with prediction len {lag}.')