#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised Autoencoder + MLP for time-series forecasting (regression), 
integrated into your existing code structure. We do a loop over forecast_windows 
and train on the specified features for each horizon.
"""

import os
import sys
os.chdir('/home/gaen/Documents/codespace-gaen/Ts-master')  # as requested
sys.path.append(os.getcwd())

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pytorch_spiking
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

##############################################################################
# 1) CONFIG
##############################################################################
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
        "color_pred_test": "#FF4136",
    },
    "data": {
        "train_split_size": 0.80,
        "input_window": 30,
        "output_window": 10,
        "train_batch_size": 256,
        "eval_batch_size": 1,
        "scaler": "normal"
    },
    "model_transformer": {
        "feature_size": 250,
        "nhead": 10,
        "num_layers": 2,
        "dropout": 0.2,
        "out_features": 1,
        "init_range": 2,
        "lr": 0.0002,
        "loss": "dilate"
    },
    "paths": {
        "drive": {
            "agg_trade": {
                "train": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
                "test": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
            },
            "orderbook": {
                "train": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
                "test": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
            },
            "models": "/content/drive/MyDrive/IP/Repos/HFTransformer/models/",
            "figures": "/content/drive/MyDrive/IP/Repos/HFTransformer/figures/",
            "utils": "/content/drive/MyDrive/IP/Repos/HFTransformer/utils/",
        },
        "local": {
            "agg_trade": {
                "train": "./data/input_data/",
                "test": "./data/input_data/",
            },
            "orderbook": {
                "train": "./data/input_data/",
                "test": "./data/input_data/",
            },
            "models": "./models/",
            "figures": "./figures/",
        },
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

################################################################################
# 2) DATA AUGMENTATION (existing logic)
################################################################################
def augment_trade_data(df, lag, forecast_window=None):
    """
    Augment input data with a 'lag_return' (for forecast_window),
    typically used as the future target.
    """
    if forecast_window:
        # Example: a lag_return for the forecast window
        df["lag_return"] = np.log(
            df["price"].shift(forecast_window) / df["price"].shift(forecast_window + 1)
        )
        return df.iloc[forecast_window + 1 :, :]
    if lag == 0:
        return df
    else:
        col_name = "log_lag" + str(lag) + "_price"
        df[col_name] = np.log(df.price) - np.log(df.price).shift(lag)
        return df.iloc[lag:, :]

################################################################################
# 3) DATASET
################################################################################
class TimeSeriesDataset(Dataset):
    """
    Convert windowed time-series data into (input, target) pairs.
    """
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

################################################################################
# 4) PREPARE DATA (same logic as your code)
################################################################################
def prepare_data_x(data, window_size, lag):
    n_row = data.shape[0] - window_size + 1
    output = np.zeros([n_row, window_size, data.shape[1]])
    x_mean = np.zeros([n_row, data.shape[1]])
    x_std = np.zeros([n_row, data.shape[1]])

    for idx in range(n_row):
        subset = data[idx : idx + window_size]
        subset_mean = np.mean(subset, axis=0)
        subset_std  = np.std(subset, axis=0) + 1e-3
        subset_norm = (subset - subset_mean) / subset_std

        x_mean[idx, :] = subset_mean
        x_std[idx, :] = subset_std
        output[idx, :, :] = subset_norm

    return output[:-lag - 1], output[-1], x_mean, x_std


def prepare_data_y(x, window_size, lag):
    output = np.zeros([len(x) - window_size - lag])
    std = 1.1 * np.sqrt(lag) + lag * 0.01
    for idx in range(len(x) - window_size - lag):
        output[idx] = np.log(
            x[window_size + lag - 1 + idx, 0] / x[window_size - 1 + idx, 0]
        ) * 10000
    output = output / std
    return output


def prepare_data(
    normalized_prices_train, dates_train, normalized_prices_test, dates_test, config, 
    lag=1, plot=False
):
    forecast_history = 100
    data_x, data_x_unseen, x_mean, x_std = prepare_data_x(
        normalized_prices_train, window_size=forecast_history, lag=lag
    )
    data_y = prepare_data_y(
        normalized_prices_train, window_size=forecast_history, lag=lag
    )

    split_index = int(data_y.shape[0] * 0.8)
    data_x_train = data_x[:split_index]
    data_y_train = data_y[:split_index]
    data_x_val   = data_x[split_index:]
    data_y_val   = data_y[split_index:]

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val

################################################################################
# 5) SUPERVISED AUTOENCODER (REGRESSION)
################################################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to the input only in training mode.
    """
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        else:
            return x

class SupervisedAutoencoderMLP(nn.Module):
    """
    1) (GaussianNoise) => Encoder => latent
    2) Decoder => reconstruct input
    3) MLP => final regression output
    4) We do MSE for both reconstruction & supervised objective
    """
    def __init__(
        self,
        input_size=38,
        latent_dim=16,
        hidden_dims_enc=[64, 32],
        hidden_dims_dec=[32, 64],
        hidden_dims_mlp=[64, 32],
        dropout=0.2,
        noise_std=0.1,
        concat_original=True,
        use_batchnorm=True,
        use_swish=True
    ):
        super().__init__()
        self.concat_original = concat_original

        # Noise
        self.noise = GaussianNoise(std=noise_std)

        # Encoder
        enc_layers = []
        in_dim = input_size
        for h in hidden_dims_enc:
            enc_layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(h))
            if use_swish:
                enc_layers.append(Swish())
            else:
                enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in hidden_dims_dec:
            dec_layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                dec_layers.append(nn.BatchNorm1d(h))
            if use_swish:
                dec_layers.append(Swish())
            else:
                dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_size))
        self.decoder = nn.Sequential(*dec_layers)

        # MLP
        mlp_layers = []
        if concat_original:
            mlp_in = latent_dim + input_size
        else:
            mlp_in = latent_dim

        for h in hidden_dims_mlp:
            mlp_layers.append(nn.Linear(mlp_in, h))
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(h))
            if use_swish:
                mlp_layers.append(Swish())
            else:
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_in = h

        mlp_layers.append(nn.Linear(mlp_in, 1))  # regression => 1 dim
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, return_reconstruction=False):
        # x can be [B, L, F]. We'll mean across L for an MLP.
        if x.ndim == 3:
            B, L, F = x.shape
            x_mean = x.mean(dim=1)  # => [B, F]
        else:
            x_mean = x

        # Noise
        x_noisy = self.noise(x_mean)

        # Encode
        z = self.encoder(x_noisy)

        # Decode
        x_recon = self.decoder(z)

        # MLP for final regression
        if self.concat_original:
            mlp_input = torch.cat([x_mean, z], dim=-1)  # => [B, input_size + latent_dim]
        else:
            mlp_input = z

        y_pred = self.mlp(mlp_input).squeeze(-1)  # => [B]
        if return_reconstruction:
            return y_pred, x_recon, z
        else:
            return y_pred

################################################################################
# 6) Factory
################################################################################
def get_model(
    model_name: str,
    input_size=38,
    **kwargs
):
    name = model_name.lower()
    if name == "supervised_autoencoder_mlp":
        return SupervisedAutoencoderMLP(
            input_size=kwargs.get("input_size", 38),
            latent_dim=kwargs.get("latent_dim", 16),
            hidden_dims_enc=kwargs.get("hidden_dims_enc", [64, 32]),
            hidden_dims_dec=kwargs.get("hidden_dims_dec", [32, 64]),
            hidden_dims_mlp=kwargs.get("hidden_dims_mlp", [64, 32]),
            dropout=kwargs.get("dropout", 0.2),
            noise_std=kwargs.get("noise_std", 0.1),
            concat_original=kwargs.get("concat_original", True),
            use_batchnorm=kwargs.get("use_batchnorm", True),
            use_swish=kwargs.get("use_swish", True)
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

################################################################################
# 7) Training function for supervised autoencoder (regression)
################################################################################
def train_autoencoder_regression(
    model,
    data_loader,
    opt,
    num_epochs=30,
    alpha=1.0,  # weight for recon
    beta=1.0,   # weight for sup
    early_stop_patience=5,
    scheduler=None,
    warmup_scheduler=None,
    device="cpu"
):
    criterion = nn.MSELoss()
    best_mse = float("inf")
    no_improve = 0
    best_state = None

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_sup = 0.0

        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opt.zero_grad()
            y_pred, x_recon, _ = model(x_batch, return_reconstruction=True)

            # (a) Reconstruction MSE vs the mean of x_batch if 3D
            if x_batch.ndim == 3:
                x_mean = x_batch.mean(dim=1)
            else:
                x_mean = x_batch
            loss_recon = criterion(x_recon, x_mean)

            # (b) Supervised MSE
            loss_sup = criterion(y_pred, y_batch)

            # (c) combined
            loss = alpha * loss_recon + beta * loss_sup
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_sup   += loss_sup.item()

        avg_loss = total_loss / len(data_loader)
        avg_recon = total_recon / len(data_loader)
        avg_sup   = total_sup   / len(data_loader)

        # Early stop on supervised MSE
        if avg_sup < best_mse:
            best_mse = avg_sup
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (MSE not improving).")
                break

        # LR schedule
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if scheduler:
                    scheduler.step()
        else:
            if scheduler:
                scheduler.step()

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Sup={avg_sup:.4f}"
        )

    if best_state:
        model.load_state_dict(best_state)
    return model

################################################################################
# 8) Evaluate R2
################################################################################
def evaluate_r2(model, data_loader, device="cpu"):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return r2_score(trues, preds), preds, trues

################################################################################
# 9) MAIN
################################################################################
if __name__ == "__main__":
    ############################################################################
    # Example reading CSV
    ############################################################################
    # In your real code, you load the actual data: e.g.:
    # agg_trade = pd.read_csv(...)
    # For demonstration, let's mock up an agg_trade DataFrame with .datetime
    size_data = 1_500_000
    data_ = {
        "datetime": pd.date_range("2020-01-01", periods=size_data, freq="S"),
        "price": np.linspace(100, 200, size_data),
        "ask1": np.random.rand(size_data),
        "askqty1": np.random.rand(size_data),
        "bid1": np.random.rand(size_data),
        "bidqty1": np.random.rand(size_data),
        "bid2": np.random.rand(size_data),
        "bidqty2": np.random.rand(size_data),
        "bid3": np.random.rand(size_data),
        "bidqty3": np.random.rand(size_data),
        "bid4": np.random.rand(size_data),
        "bidqty4": np.random.rand(size_data),
        "bid5": np.random.rand(size_data),
        "bidqty5": np.random.rand(size_data),
        "bid6": np.random.rand(size_data),
        "bidqty6": np.random.rand(size_data),
        "bid7": np.random.rand(size_data),
        "bidqty7": np.random.rand(size_data),
        "bid8": np.random.rand(size_data),
        "bidqty8": np.random.rand(size_data),
        "bid9": np.random.rand(size_data),
        "bidqty9": np.random.rand(size_data),
        "ask2": np.random.rand(size_data),
        "askqty2": np.random.rand(size_data),
        "ask3": np.random.rand(size_data),
        "askqty3": np.random.rand(size_data),
        "ask4": np.random.rand(size_data),
        "askqty4": np.random.rand(size_data),
        "ask5": np.random.rand(size_data),
        "askqty5": np.random.rand(size_data),
        "ask6": np.random.rand(size_data),
        "askqty6": np.random.rand(size_data),
        "ask7": np.random.rand(size_data),
        "askqty7": np.random.rand(size_data),
        "ask8": np.random.rand(size_data),
        "askqty8": np.random.rand(size_data),
        "ask9": np.random.rand(size_data),
        "askqty9": np.random.rand(size_data),
    }
    agg_trade = pd.DataFrame(data_)

    ############################################################################
    # Forecast windows
    ############################################################################
    forecast_windows = [1, 2, 3, 4, 5]  # or up to 32, etc.

    for forecast_window in forecast_windows:
        print("\n---------------------------------------------")
        print(f"Forecast window = {forecast_window}")
        print("---------------------------------------------")

        # augment data
        orderbook = augment_trade_data(agg_trade.copy(), lag=0, forecast_window=forecast_window)

        # features
        features = [
            "price",
            "lag_return",
            "bid1",
            "bidqty1",
            "bid2",
            "bidqty2",
            "bid3",
            "bidqty3",
            "bid4",
            "bidqty4",
            "bid5",
            "bidqty5",
            "bid6",
            "bidqty6",
            "bid7",
            "bidqty7",
            "bid8",
            "bidqty8",
            "bid9",
            "bidqty9",
            "ask1",
            "askqty1",
            "ask2",
            "askqty2",
            "ask3",
            "askqty3",
            "ask4",
            "askqty4",
            "ask5",
            "askqty5",
            "ask6",
            "askqty6",
            "ask7",
            "askqty7",
            "ask8",
            "askqty8",
            "ask9",
            "askqty9",
        ]
        orderbook = orderbook.dropna()  # ensure no NaNs
        if len(orderbook) < 1_250_000:
            print("Not enough data for this example. Skipping.")
            continue

        # Example train/validation split from your snippet
        data_x_train, data_y_train, data_x_val, data_y_val = None, None, None, None
        split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
            np.array(orderbook[features][1_000_000:1_250_000]),  # train portion
            np.array(agg_trade.datetime[899_999:1_000_000]),     # unused in example
            np.array(orderbook[features][60_000:60_600]),        # unused test portion
            np.array(agg_trade.datetime[60_000:60_600]),         # unused in example
            config,
            lag=forecast_window,
            plot=False,
        )

        train_dataset = TimeSeriesDataset(data_x_train, data_y_train)
        val_dataset   = TimeSeriesDataset(data_x_val, data_y_val)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Build the model
        model_name = "supervised_autoencoder_mlp"
        model_custom = get_model(
            model_name=model_name,
            input_size=len(features),
            latent_dim=16,
            hidden_dims_enc=[64, 64],
            hidden_dims_dec=[64, 64],
            hidden_dims_mlp=[64, 32],
            dropout=0.2,
            noise_std=0.1,
            concat_original=True,
            use_batchnorm=True,
            use_swish=True
        ).to(device)

        optimizer = optim.Adam(model_custom.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=100)

        # Train
        model_custom = train_autoencoder_regression(
            model=model_custom,
            data_loader=train_loader,
            opt=optimizer,
            num_epochs=30,
            alpha=1.0,         # weight for reconstruction
            beta=1.0,          # weight for supervised MSE
            early_stop_patience=5,
            scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,
            device=device
        )

        # Evaluate on validation
        r2_val, preds_val, trues_val = evaluate_r2(model_custom, val_loader, device=device)
        print(f"Validation R2 for horizon={forecast_window} => {r2_val:.4f}")

        # Save final model
        save_dir = os.path.join(
            "/home/gaen/Documents/codespace-gaen/Ts-master/playround_models/",
            model_name
        )
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = os.path.join(
            save_dir,
            f"{model_name}_forecast_{forecast_window}.pt",
        )
        torch.save(model_custom.state_dict(), final_model_path)
        print(f"Saved final model to: {final_model_path}")

    print("All done!")
