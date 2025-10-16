#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refactored script for time-series forecasting using S4 and Transformer models.
"""

from pathlib import Path
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pytorch_spiking
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Import from local modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from models_s4.s4.s4 import S4Block as S4
from models_s4.s4.s4d import S4D
from trans4trade.models import get_model
from trans4trade.helpers import save_df, save_to_log

################################################################################
# CONFIG
################################################################################
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
        "init_range": 2,  # 0.5
        "lr": 0.0002,     # 0.0001
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

################################################################################
# DEVICE CONFIG & MISC
################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
drive = False
print(f"Using device: {device}")

################################################################################
# DATA AUGMENTATION
################################################################################
def augment_trade_data(df, lag, forecast_window=None):
    """
    Augment input data with lagged returns or forecast window returns.
    """
    if forecast_window:
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
# DATALOADER
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


def prepare_data_x(data, window_size, lag):
    """
    Window the input data for ML models.
    """
    n_row = data.shape[0] - window_size + 1
    if ttt == 1: 
        save_to_log(data.shape, "data shape")
        save_to_log(n_row, "n_row")

    subset_mean = np.mean(data[:window_size], axis=0)
    output = np.zeros([n_row, window_size, len(subset_mean)])
    x_mean = np.zeros([n_row, len(subset_mean)])
    x_std = np.zeros([n_row, len(subset_mean)])

    for idx in range(n_row):
        subset = data[idx : idx + window_size]
        #save_to_log(subset, "subset")
        subset_mean = np.mean(subset, axis=0)
        #save_to_log(subset_mean, "subset_mean")
        subset_std = np.std(subset, axis=0) + 0.01

        subset_norm = (subset - subset_mean) / subset_std
        x_mean[idx, :] = subset_mean
        x_std[idx, :] = subset_std
        output[idx, :, :] = subset_norm
    return output[:-lag - 1], output[-1], x_mean, x_std


def prepare_data_y(x, window_size, lag):
    """
    Window the target data for the ML models.
    """
    output = np.zeros([len(x) - window_size - lag])
    std = 1.1 * np.sqrt(lag) + lag * 0.01

    for idx in range(len(x) - window_size - lag):
        output[idx] = np.log(x[window_size + lag - 1 + idx, 0] / x[window_size - 1 + idx, 0]) * 10000

    output = output / std
    return output

ttt = 0
def prepare_data(normalized_prices_train, dates_train, normalized_prices_test, dates_test, config, lag=1, plot=False):
    """
    Wrapper to return train/val splits. (Currently returns data_x_train, data_y_train, etc.)
    """
    data_x, data_x_unseen, x_mean, x_std = prepare_data_x(normalized_prices_train, window_size=forecast_history, lag=lag)
    data_y = prepare_data_y(normalized_prices_train, window_size=forecast_history, lag=lag)

    split_index = int(data_y.shape[0] * 0.8)
    data_x_train = data_x[:split_index]
    if ttt == 1 :
        save_to_log(data_x_train.shape,"data_x_train")
        save_df(data_x_train[0],"data_x_train")
        xxx
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val

################################################################################
# CUSTOM LOSSES
################################################################################
def quantile_loss(y, y_pred, quantile):
    """
    Standard quantile loss as in TFT paper.
    """
    if quantile < 0 or quantile > 1:
        raise ValueError(f"Illegal quantile={quantile}! Should be between 0 and 1.")

    prediction_underflow = y - y_pred
    q_loss = (
        quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow))
        + (1.0 - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))
    )
    return torch.sum(q_loss, axis=-1)

################################################################################
# HELPER FUNCTIONS
################################################################################
criterion_dict = {
    "MAE": nn.L1Loss,
    "MSE": nn.MSELoss,
    "QuantileLoss": quantile_loss,
}


def compute_loss(labels, output, src, criterion):
    """
    Compute a single loss value.
    """
    if isinstance(output, torch.Tensor):
        if len(labels.shape) != len(output.shape):
            # Broadcasting / shaping
            if len(labels.shape) > 1 and (labels.shape[1] == output.shape[1]):
                labels = labels.unsqueeze(2)
            else:
                labels = labels.unsqueeze(0)

    loss = criterion(output, labels.float())
    return loss


def train_step(
    model, opt, criterion, data_loader, takes_target, device, num_targets=1, forward_params={}
):
    """
    Perform training over one epoch.
    """
    i = 0
    running_loss = 0.0
    model.train()

    for src, trg in data_loader:
        opt.zero_grad()
        if takes_target:
            forward_params["t"] = trg.to(device)

        src = src.to(device)
        trg = trg.to(device)
        output = model(src, **forward_params)
        output = output.squeeze()

        if num_targets == 1:
            labels = trg
        else:
            labels = trg[:, :, 0 : num_targets]

        loss = compute_loss(labels, output, src, criterion[0])
        loss.backward()
        opt.step()

        running_loss += loss.item()
        i += 1

    return running_loss


def validation(val_loader, model, criterion, device, num_targets=1):
    """
    Compute validation loss & gather predictions.
    """
    crit_losses = dict.fromkeys(criterion, 0)
    model.eval()
    labels_all = torch.Tensor(0).to(device)
    output_all = torch.Tensor(0).to(device)

    with torch.no_grad():
        for src, targ in val_loader:
            src = src.to(device)
            targ = targ.to(device)
            output = model(src.float()).squeeze()
            output_all = torch.cat((output_all, output))

            if num_targets == 1:
                labels = targ
            else:
                labels = targ[:, :, 0 : num_targets]

            for crit in criterion:
                loss = compute_loss(labels, output, src, crit)
                crit_losses[crit] += loss.item()

            labels_all = torch.cat((labels_all, labels))

    return list(crit_losses.values())[0], output_all, labels_all


def forecast(data_loader, model, criterion, forecast_horizon, device, num_targets=1):
    """
    Forecast for certain steps in the data_loader.
    """
    crit_losses = dict.fromkeys(criterion, 0)
    model.eval()
    output_decoder = torch.Tensor(0).to(device)
    labels_all = torch.Tensor(0).to(device)

    counter = 0
    with torch.no_grad():
        for src, targ in data_loader:
            if (counter % forecast_horizon) == 0:
                src = src.to(device)

                targ = targ.to(device)
                output = model(src.float())
                output_decoder = torch.cat((output_decoder, output))

                if num_targets == 1:
                    labels = targ
                else:
                    labels = targ[:, :, 0 : num_targets]

                for crit in criterion:
                    loss = compute_loss(labels, output, src, crit)
                    crit_losses[crit] += loss.item()

                labels_all = torch.cat((labels_all, labels))
            counter += 1

    return list(crit_losses.values())[0], output_decoder, labels_all

################################################################################
# TRAINER
################################################################################
def strategy_evaluator(true, pred):
    """
    Evaluates strategy in terms of correct buys, sells, and holds.
    """
    total_buys = np.sum(true > 0)
    total_sells = np.sum(true < 0)
    total_holds = np.sum(true == 0)
    total_correct_buys = total_correct_sells = total_correct_holds = 0

    for idx in range(len(true)):
        for jdx in range(len(true[0])):
            if true[idx, jdx] > 0 and pred[idx, jdx] > 0:
                total_correct_buys += 1
            elif true[idx, jdx] < 0 and pred[idx, jdx] < 0:
                total_correct_sells += 1
            elif true[idx, jdx] == 0 and pred[idx, jdx] == 0:
                total_correct_holds += 1

    total_correct_buys_r = total_correct_buys / total_buys if total_buys != 0 else 0
    total_correct_sells_r = total_correct_sells / total_sells if total_sells != 0 else 0
    total_correct_holds_r = total_correct_holds / total_holds if total_holds != 0 else 0

    return (
        round(total_correct_buys_r, 3),
        round(total_correct_sells_r, 3),
        round(total_correct_holds_r, 3),
    )


def trainer(
    model,
    train_loader,
    validation_loader,
    test_loader,
    criterion,
    opt,
    scheduler,
    warmup_scheduler,
    max_epochs,
    batch_size,
    forecast_horizon,
    takes_target,
    shuffle=False,
    num_targets=1,
    plot_prediction=True,
    save_path=None,
    LAG=0,
):
    """
    Main training routine.
    """
    start_time = time.time()

    data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=False, num_workers=10)
    validation_data_loader = DataLoader(validation_loader, batch_size=batch_size, shuffle=False, num_workers=10)
    test_data_loader = None  # Provided but not used
    forecast_data_loader = DataLoader(validation_loader, batch_size=1, shuffle=False, num_workers=2)

    for epoch in range(1, max_epochs + 1):
        total_loss = train_step(
            model, opt, criterion, data_loader, takes_target, device, num_targets=num_targets
        )

        if plot_prediction:
            val_loss, val_values, true_values = forecast(
                forecast_data_loader,
                model,
                criterion,
                forecast_horizon=forecast_horizon,
                device=device,
                num_targets=num_targets,
            )
            # Optionally show or save plot
            fig, ax = plt.subplots(1, 1, figsize=(18, 8))
            ax.plot(true_values.cpu().view(-1), label="truth", alpha=0.3)
            ax.plot(val_values.cpu().view(-1), label="forecast", alpha=0.8)
            ax.set_xlim(left=0, right=len(true_values.cpu().view(-1)))
            plt.legend()
            plt.show()
        else:
            val_loss, val_values, true_values = validation(
                validation_data_loader, model, criterion, device, num_targets=num_targets
            )

        preds = val_values.cpu().numpy()
        trues = true_values.cpu().numpy()

        r2_sklearn = r2_score(trues, preds)
        elapsed = time.time() - start_time

        print("-" * 88)
        print(
            f"| epoch {epoch:3d} | {elapsed:5.2f}s | "
            f"train loss {total_loss:5.5f} | val loss {val_loss:5.5f} | "
            f"lr {scheduler.get_last_lr()[0]:1.8f} | r2 sklearn: {r2_sklearn:1.5f} |"
        )
        print("-" * 88)
        start_time = time.time()

        # Save intermediate results
        if save_path:
            results = {
                "model": model_name,
                "pred_len": forecast_horizon,
                "epoch": epoch,
                "train_loss": total_loss,
                "val_loss": val_loss,
                "r2_val_sklearn": r2_sklearn,
            }
            # Append results to CSV
            import csv

            csv_file = os.path.join(save_path, "results.csv")
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(results)

            # Save model if r2 is good
            save_directory = os.path.join(save_path, model_name)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            if r2_sklearn > 0.02:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_directory,
                        f"_epoch_{epoch}_time_{int(time.time())}_r2_{r2_sklearn}.pt",
                    ),
                )

        with warmup_scheduler.dampening():
            scheduler.step()

################################################################################
# EXAMPLE MAIN (COMMENT/EDIT AS NEEDED)
################################################################################
if __name__ == "__main__":
    # Example usage:
    date_train = "All_to_Sept"
    date_test = "All_to_Sept"

    # Example reading CSV
    agg_trade = pd.read_csv(
        config["paths"]["local"]["agg_trade"]["train"] + date_train + "/orderbook_agg_trade_dollarvol.csv"
        #config["paths"]["local"]["agg_trade"]["train"] + date_train + "/orderbook.csv"
        )
    agg_trade["price"] = (
        agg_trade["ask1"] * agg_trade["askqty1"]
        + agg_trade["bid1"] * agg_trade["bidqty1"]
    ) / (agg_trade["askqty1"] + agg_trade["bidqty1"])

    model_name = "lstm_attn"   # or s4transformer, lstm, deeplob, lstm_attn, vanilla_transformer, stacklstm, trans_enc_lstm
    base_save_path = os.path.join(
        "/home/gaen/Documents/codespace-gaen/Ts-master/playround_models/",
        model_name,
        "training_details/"
    )
    os.makedirs(base_save_path, exist_ok=True)

    forecast_history = 1200
    epochs = 30
    batch_size = 64
    forecast_windows = [i for i in range(1, 32)]

    for forecast_window in forecast_windows:
        orderbook = augment_trade_data(agg_trade, lag=0, forecast_window=forecast_window)
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

        # Example train/validation split
        split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
            np.array(orderbook[features][1_000_000:1_040_000]),
            np.array(agg_trade.datetime[899_999:1_000_000]),  # unused in example
            np.array(orderbook[features][60_000:60_600]),
            np.array(agg_trade.datetime[60_000:60_600]),      # unused in example
            config,
            lag=forecast_window,
            plot=False,
        )

        train_loader = TimeSeriesDataset(data_x_train, data_y_train)
        val_loader = TimeSeriesDataset(data_x_val, data_y_val)
        test_loader = None  # Unused

        model_custom = get_model(
            model_name=model_name,
            input_size=38,
            output_size=1,
            #d_model=forecast_history*0.8,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.1,
            trans_lstm_hidden=forecast_history*0.8,
            trans_lstm_layers=1
        ).to(device)

        criterion = nn.MSELoss(reduction="sum")
        optimizer = optim.AdamW(model_custom.parameters(), lr=0.1, amsgrad=True)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=1000)

        trainer(
            model_custom,
            train_loader,
            val_loader,
            test_loader,
            [criterion],
            optimizer,
            scheduler,
            warmup_scheduler,
            epochs,
            batch_size=batch_size,
            forecast_horizon=forecast_window,
            takes_target=False,
            plot_prediction=False,
            save_path=base_save_path,
            LAG=forecast_window,
        )

        # Free memory
        del data_x_train
        del data_y_train
        del data_x_val
        del data_y_val

        # Save final model
        final_model_path = os.path.join(
            "/home/gaen/Documents/codespace-gaen/Ts-master/playround_models/",
            model_name,
            f"{model_name}_forecasting_FINAL_horizon_{forecast_window}.pt",
        )
        torch.save(model_custom, final_model_path)
        print(f"Done with prediction len {forecast_window}.")
