import os
from datetime import datetime
import time
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        "USE_DRIVE": False,
        "TRAIN_DATE": "All_to_Sept",
        "FILE_NAME": "orderbook_agg_trade_dollarvol.csv",
        "DRIVE_FILE_NAME": "orderbook.csv"  
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
    TRAIN_SIZE = 600000   # Number of rows for training
    VAL_SIZE = 10000       # Number of rows for validation
    WINDOW_SIZE = 100
    LAG = 1
    # Remove or ignore TRAIN_SPLIT to prevent double-splitting
    FORECAST_WINDOWS = list(range(1, 32))  # Forecast windows
    
    # Model Architecture
    INPUT_SIZE = len(FEATURES)
    HIDDEN_SIZE = 32
    NUM_LSTM_LAYERS = 5
    DROPOUT_RATE = 0.25
    OUTPUT_SIZE = 1
    
    # These are not used in the example but can be relevant if you implement an explicit attention layer:
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
    """
    Model combining a Transformer Encoder (which provides self-attention)
    and an LSTM stack, for time series prediction.
    """
    def __init__(
        self, 
        input_size=Config.INPUT_SIZE, 
        hidden_layer_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS, 
        output_size=Config.OUTPUT_SIZE,
        dropout=Config.DROPOUT_RATE,
        nhead=4
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        # Linear projection of inputs
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.prelu = nn.PReLU()
        
        # Transformer Encoder
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
        self.lstm = nn.LSTM(
            hidden_layer_size, 
            hidden_size=hidden_layer_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_layer_size)
        self.layer_norm2 = nn.LayerNorm(hidden_layer_size)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_size]
        """
        # Project input
        x_proj = self.linear_1(x)
        x_proj = self.prelu(x_proj)
        
        # Transformer encoder (with skip connection)
        x_norm = self.layer_norm1(x_proj)
        x_perm = x_norm.permute(1, 0, 2)    # -> [seq_len, batch_size, hidden_size]
        transformer_out = self.transformer_encoder(x_perm)
        transformer_out = transformer_out.permute(1, 0, 2)  
        # Add skip/residual from x_proj
        transformer_out = transformer_out + x_proj
        
        # LSTM
        lstm_out, _ = self.lstm(transformer_out)  
        lstm_out = self.layer_norm2(lstm_out)
        
        # Take last time step
        last_step = lstm_out[:, -1, :]  
        
        # Final projection
        out = self.dropout(last_step)
        out = self.linear_2(out)
        
        return out

    def train_model(
        self, 
        train_dataloader, 
        val_dataloader, 
        learning_rate, 
        scheduler_step_size,
        n_epochs=50, 
        device="cpu", 
        save_path=None, 
        forecast_window=None
    ):
        """
        Train the model on the given data.
        """

        def run_epoch(dataloader, is_training=False):
            epoch_loss = 0
            outputs = []
            targets = []
            if is_training:
                self.train()
            else:
                self.eval()
                
            for (x_batch, y_batch) in dataloader:
                if is_training:
                    optimizer.zero_grad()

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).reshape(-1, 1)

                out = self(x_batch)
                loss = criterion(out, y_batch)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                outputs.append(out.detach().cpu())
                targets.append(y_batch.detach().cpu())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            r2 = r2_score(targets.numpy(), outputs.numpy())
            return epoch_loss / len(dataloader), r2

        # Optimizer, scheduler, and loss
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=Config.SCHEDULER_GAMMA)

        # Training Loop
        for epoch in range(n_epochs):
            train_loss, r2_train = run_epoch(train_dataloader, is_training=True)
            val_loss, r2_val = run_epoch(val_dataloader, is_training=False)
            scheduler.step()

            if save_path:
                results = {
                    'model': 'LSTM_MO',
                    'pred_len': forecast_window,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'r2_val_sklearn': r2_val
                }
                df = pd.DataFrame([results])
                df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

            print(
                f'Epoch[{epoch+1}/{n_epochs}] | train_loss:{train_loss:.6f}, '
                f'val_loss:{val_loss:.6f} | lr:{scheduler.get_last_lr()[0]:.6f} | r2_val: {r2_val:.5f}'
            )



#####################################
#           DATA PREPARATION        #
#####################################

def prepare_data_x(data, window_size, lag):
    """
    Build overlapping windows of shape [window_size, n_features].
    Returns:
      - 3D array [num_samples, window_size, n_features]
      - x_mean, x_std for possible inverse transformations (if needed)
    """
    # Adjust length to avoid indexing past the end
    # We subtract (window_size + lag - 1) from total rows
    n_row = data.shape[0] - (window_size + lag) + 1
    if n_row <= 0:
        raise ValueError("Not enough data for the specified window_size and lag.")
    
    output = []
    x_means = []
    x_stds = []
    for i in range(n_row):
        subset = data[i : i + window_size]
        mu = subset.mean(axis=0)
        sigma = subset.std(axis=0) + 1e-5
        # normalized subset
        subset_norm = (subset - mu) / sigma
        output.append(subset_norm)
        x_means.append(mu)
        x_stds.append(sigma)

    return np.array(output), np.array(x_means), np.array(x_stds)

def prepare_data_y(prices, window_size, lag):
    """
    Example target building function:
    We take log returns over [window_size-1+lag, window_size-1].
    Adjust to your actual target definition.
    """
    y_list = []
    # We can produce the same number of outputs as in X.
    # That is len(prices) - (window_size + lag) + 1
    n_row = len(prices) - (window_size + lag) + 1
    
    # Example: log return * 10,000 with some scaling
    for i in range(n_row):
        start_idx = i + window_size - 1
        end_idx = start_idx + lag
        # log-return from price[start_idx] -> price[end_idx]
        ret = np.log(prices[end_idx] / prices[start_idx]) * 10_000
        y_list.append(ret)

    return np.array(y_list)

def prepare_data(data_array, config, lag=1):
    """
    Prepare X, y from the input data for a single forecast window.
    This version DOES NOT do an internal train/val split by ratio.
    We simply return X, y for the entire slice so that 
    the calling code can do its own slicing (the code outside 
    has already chosen the data slice).
    """
    window_size = config["data"]["window_size"]

    data_x, x_means, x_stds = prepare_data_x(data_array, window_size, lag)
    # For the target, let's assume the first column is 'price':
    # e.g. data_array[:,0] if 'price' is the first column
    y_vals = prepare_data_y(data_array[:, 0], window_size, lag)

    return data_x, y_vals, x_means, x_stds


#####################################
#           DATA LOADING            #
#####################################

def preprocess_data(df, forecast_window):
    """Preprocess data by adding lag_return and dropping initial NaNs."""
    df = df.copy()
    # Example: shift price by forecast_window steps
    df['lag_return'] = np.log(
        df['price'].shift(forecast_window) / df['price'].shift(forecast_window + 1)
    )
    # Drop those rows
    df = df.iloc[forecast_window+1:].reset_index(drop=True)
    return df

def load_data():
    """Load raw data using config path."""
    return pd.read_csv(Config.get_data_path())


#####################################
#           MODEL TRAINING          #
#####################################

# Create directory for model saves
Config.create_model_save_dir()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load entire dataset
agg_trade = load_data()

for lag in Config.FORECAST_WINDOWS:
    # 1) Preprocess for this forecast window
    orderbook = preprocess_data(agg_trade, lag)
    
    # 2) Slice the data externally, so no ratio-splitting is done inside
    #    Train data slice:
    train_df = orderbook.iloc[:Config.TRAIN_SIZE]
    #    Validation data slice:
    val_df = orderbook.iloc[Config.TRAIN_SIZE : Config.TRAIN_SIZE + Config.VAL_SIZE]

    # 3) Prepare X, y for train
    train_X_full, train_y_full, _, _ = prepare_data(
        train_df[Config.FEATURES].values, 
        Config.get_config_dict(), 
        lag=lag
    )

    # 4) Prepare X, y for val
    val_X_full, val_y_full, _, _ = prepare_data(
        val_df[Config.FEATURES].values, 
        Config.get_config_dict(), 
        lag=lag
    )

    # 5) Build datasets
    train_dataset = TimeSeriesDataset(train_X_full, train_y_full)
    val_dataset = TimeSeriesDataset(val_X_full, val_y_full)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=Config.BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=Config.BATCH_SIZE,
                                shuffle=False)

    # 6) Define model
    model_MO = LSTM_MO(
        input_size=Config.INPUT_SIZE,
        hidden_layer_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout=Config.DROPOUT_RATE
    ).to(device)

    # 7) Train
    model_MO.train_model(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        learning_rate=Config.LEARNING_RATE,
        scheduler_step_size=Config.SCHEDULER_STEP, 
        n_epochs=Config.NUM_EPOCHS,
        device=device, 
        save_path=os.path.join(
            Config.BASE_DIR, 
            Config.MODEL_SAVE_PATH, 
            f"{int(time.time())}_results.csv"
        ), 
        forecast_window=lag
    )

    # 8) Save model
    model_save_folder = os.path.join(
        Config.BASE_DIR,
        Config.MODEL_SAVE_PATH,
        "No.2"
    )
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    torch.save(model_MO, f"{model_save_folder}/LSTM_MO_LAG_{lag}.pt")
    print(f"Done with prediction length {lag}.")
