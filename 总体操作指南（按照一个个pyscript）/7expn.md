
---

### 1. **安装必要的库和设置工作目录**

```python
# %%
!pip --quiet install pytorch-warmup  # 静默安装pytorch-warmup库，用于实现学习率预热
%cd /home/gaen/Documents/codespace-gaen/Simons  # 更改当前工作目录到指定路径
# 这个cd命令将工作目录切换到你家里主机上的项目目录
```

**说明：**
- 使用`pip`命令安装`pytorch-warmup`库，该库有助于实现学习率的预热策略，改善模型训练的稳定性和效果。
- 使用`%cd`命令更改当前工作目录到指定的路径，确保后续的文件操作在正确的目录下进行。

---

### 2. **安装并导入必要的库**

```python
# %%
from IPython.display import clear_output 
!pip --quiet install pytorch_spiking pytorch_lightning  # 静默安装pytorch_spiking和pytorch_lightning库
# pip --quiet install pytorch_forecasting 
clear_output()  # 清除输出，保持Notebook整洁

# %%
import os
import time
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import pickle

import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pytorch_spiking
import pytorch_warmup as warmup

import matplotlib.pyplot as plt
```

**说明：**
- 安装并导入多个关键的Python库，包括：
  - **数据处理与分析**：`numpy`、`pandas`
  - **机器学习评估**：`sklearn.metrics`中的`r2_score`用于计算R²评分
  - **深度学习框架**：`torch`及其子模块，如`nn`（神经网络模块）、`optim`（优化器）、`Dataset`和`DataLoader`（数据加载）
  - **辅助库**：`pickle`用于序列化对象，`matplotlib`用于绘图
  - **其他库**：`pytorch_spiking`用于脉冲神经网络激活函数，`pytorch_warmup`用于学习率预热策略

---

### 3. **配置参数**

```python
# %%
config = {
    "plots": {  # 绘图相关配置
        "show_plots": False,  # 是否显示图表
        "xticks_interval": 1200,  # x轴刻度间隔
        "color_actual": "#001f3f",  # 实际值颜色
        "color_train": "#3D9970",  # 训练集颜色
        "color_val": "#0074D9",  # 验证集颜色
        "color_test": "#FF4136",  # 测试集颜色
        "color_pred_train": "#3D9970",  # 训练集预测颜色
        "color_pred_val": "#0074D9",  # 验证集预测颜色
        "color_pred_test": "#FF4136",  # 测试集预测颜色
    },
    "data": {  # 数据处理相关配置
        "train_split_size": 0.80,  # 训练集划分比例
        "input_window": 150,  # 输入窗口大小
        "output_window": 50,  # 输出窗口大小
        "train_batch_size": 32,  # 训练批次大小
        "eval_batch_size": 1,  # 评估批次大小
        "scaler": "normal"  # 数据标准化方式
    },
    "model_transformer": {  # Transformer模型相关配置
        "feature_size": 250,  # 特征维度
        "nhead": 10,  # 多头注意力机制的头数
        "num_layers": 2,  # 编码器层数
        "dropout": 0.2,  # Dropout率
        "out_features": 1,  # 输出特征维度
        "init_range": 2,  # 初始化范围
        "lr": 0.0002,  # 学习率
        "loss": "dilate"  # 损失函数类型
    },
    "paths": {  # 文件路径配置
        "drive": {  # Google Drive路径配置
            "agg_trade": {  # 聚合交易数据路径
                "train": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
                "test": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
            },
            "orderbook": {  # 订单簿数据路径
                "train": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
                "test": "/content/drive/MyDrive/IP/Repos/HFTransformer/input_data/",
            },
            "models": "/content/drive/MyDrive/IP/Repos/HFTransformer/models/",  # 模型保存路径
            "figures": "/content/drive/MyDrive/IP/Repos/HFTransformer/figures/",  # 图表保存路径
            "utils": "/content/drive/MyDrive/IP/Repos/HFTransformer/utils/",  # 工具脚本路径
        },
        "local": {  # 本地路径配置
            "agg_trade": {  # 聚合交易数据路径
                "train": "./input_data/",
                "test": "./input_data/",
            },
            "orderbook": {  # 订单簿数据路径
                "train": "./input_data/",
                "test": "./input_data/",
            },
            "models": "./models/",  # 模型保存路径
            "figures": "./figures/",  # 图表保存路径
        }
    }
}
```

**说明：**
- **`config`** 字典包含了多个配置参数，用于控制不同模块的行为：
  - **绘图配置**：控制图表的显示、颜色等。
  - **数据配置**：定义数据的窗口大小、批次大小、训练集划分比例等。
  - **模型配置**：设置Transformer模型的结构参数，如特征维度、多头数、层数、Dropout率、学习率等。
  - **路径配置**：定义数据和模型在不同环境下（如Google Drive或本地）的存储路径。

---

### 4. **设置计算设备**

```python
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有可用GPU，若有则使用GPU，否则使用CPU
drive = True  # 标记是否使用Google Drive
print(device)  # 打印当前使用的设备类型
```

**说明：**
- 使用`torch.cuda.is_available()`检查是否有可用的GPU（CUDA），若有则设置设备为`cuda`，否则为`cpu`。
- 打印当前使用的设备类型，帮助确认是否成功使用了GPU加速。

---

### 5. **数据增强与准备函数**

#### a. **数据增强函数**

```python
# %%
def augment_trade_data(df, lag, forecast_window=None):
    '''
    增强输入数据。
    '''
    if forecast_window:
        df['lag_return'] = np.log(df['price'].shift(forecast_window) / df['price'].shift(forecast_window + 1))
        return df.iloc[forecast_window + 1:, :]
    if lag == 0:
        return df
    else:
        col_name = 'log_lag' + str(lag) + '_price'
        df[col_name] = np.log(df.price) - np.log(df.price).shift(lag)
        return df.iloc[lag:, :]
    
# 后续会用到，模拟了一个正常交易时候的延迟
```

**说明：**
- **`augment_trade_data`** 函数用于对原始交易数据进行增强处理，生成额外的特征：
  - 如果指定了`forecast_window`，计算未来窗口的对数收益率（`lag_return`），即未来`forecast_window`步的价格变化。
  - 如果`lag`不为零，计算价格的对数差分，模拟交易延迟。
  - 返回增强后的数据框，去除因滞后或预测窗口产生的NaN值。

#### b. **数据准备函数**

```python
def prepare_data_x(data, window_size, lag):
    '''
    将输入数据窗口化，为机器学习模型准备输入。
    '''
    n_row = data.shape[0] - window_size + 1  # 窗口化后的样本数
    subset = data[:window_size]
    subset_mean = np.mean(subset, axis=0)
    output = np.zeros([n_row, window_size, len(subset_mean)])  # 存储窗口化后的输入数据
    x_mean = np.zeros([n_row, len(subset_mean)])  # 存储每个窗口的均值
    x_std = np.zeros([n_row, len(subset_mean)])  # 存储每个窗口的标准差
    for idx in range(n_row):
        subset = data[idx:idx + window_size]
        subset_mean = np.mean(subset, axis=0)
        subset_std = np.std(subset, axis=0) + 0.01  # 防止除以零
        subset_norm = (subset - subset_mean) / subset_std  # 标准化
        x_mean[idx, :] = subset_mean
        x_std[idx, :] = subset_std
        output[idx, :, :] = subset_norm
    x_mean = np.array(x_mean)
    x_std = np.array(x_std)
    return output[:-lag - 1], output[-1], x_mean, x_std  # 返回窗口化输入数据和统计量

def prepare_data_y(x, window_size, lag):
    '''
    将目标数据窗口化，为机器学习模型准备目标。
    '''
    output = np.zeros([len(x) - window_size - lag])
    std = 1.1 * np.sqrt(lag) + lag * 0.01  # 标准化因子
    for idx in range(0, len(x) - window_size - lag):
        # 计算未来lag步的对数收益率，并放大1万倍
        output[idx] = np.log(x[window_size + lag - 1 + idx, 0] / x[window_size - 1 + idx, 0]) * 10_000
    output = output / std  # 标准化
    return output

def prepare_data(normalized_prices_train, dates_train, normalized_prices_test, dates_test, config, lag=1, plot=False):
    '''
    准备输入和目标数据，返回训练集和验证集。
    '''
    data_x, data_x_unseen, x_mean, x_std = prepare_data_x(normalized_prices_train, window_size=100, lag=lag)
    data_y = prepare_data_y(normalized_prices_train, window_size=100, lag=lag)
    split_index = int(data_y.shape[0] * 0.8)  # 80%的数据作为训练集
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val
```

**说明：**
- **`prepare_data_x`** 函数将输入数据进行窗口化处理：
  - 将时间序列数据按照固定窗口大小（`window_size`）分割成多个样本。
  - 对每个窗口的数据进行标准化（减均值，除以标准差）。
  - 返回窗口化后的输入数据、最后一个窗口的标准化数据及其均值和标准差。
  
- **`prepare_data_y`** 函数生成目标数据：
  - 计算未来`lag`步的对数收益率，放大1万倍以增加数值稳定性。
  - 对目标数据进行标准化处理。
  
- **`prepare_data`** 函数整合上述两个函数：
  - 将数据划分为训练集和验证集，比例为80%训练，20%验证。
  - 返回划分后的训练和验证数据。

**数据管道流程：**
1. **数据增强**：通过`augment_trade_data`函数生成额外的特征，如对数收益率。
2. **窗口化处理**：使用`prepare_data_x`和`prepare_data_y`函数将时间序列数据转换为模型可接受的输入（窗口化的标准化数据）和目标（未来收益率）。
3. **数据划分**：将窗口化后的数据按照80%的比例划分为训练集和验证集。

---

### 6. **定义Transformer模型**

```python
# %%
class HFformer(nn.Module):
    def __init__(self, n_time_series, seq_len, output_seq_len, d_model=128, n_heads=8,
                 n_layers_encoder=6, dropout=0.1, output_dim=1, forward_dim=2048, use_mask=False, quantiles=None):
        '''
        定义HFformer模型。包含Transformer编码器、线性解码器，并使用脉冲激活函数。
        '''
        super(HFformer, self).__init__()
        self.device = device
        self.n_time_series = n_time_series
        self.d_model = d_model
        self.nheads = n_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.n_layers_encoder = n_layers_encoder
        self.seq_len = seq_len
        self.output_seq_len = output_seq_len
        self.mask_it = use_mask
        self.quantiles = quantiles
        self.output_dim = output_dim 

        self.dense_shape = nn.Linear(self.n_time_series, self.d_model)  # 输入线性层
        spiking_activation = pytorch_spiking.SpikingActivation(nn.PReLU().to(self.device)).to(self.device)  # 脉冲激活函数

        # 定义Transformer编码器层
        self.encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads, self.forward_dim, self.dropout,
                                                    activation=spiking_activation).to(self.device)
        self.encoder_norm = LayerNorm(self.d_model)  # 层归一化
        self.transformer_enc = TransformerEncoder(self.encoder_layer, self.n_layers_encoder, self.encoder_norm).to(self.device)        
        self.output_dim_layer = nn.Linear(self.d_model, self.output_dim)  # 输出线性层
        # self.output_dim_layer = nn.LSTM(self.d_model, self.output_dim, 1, batch_first=False)
        if quantiles:
            self.out_length_lay = nn.Linear(self.seq_len, len(quantiles))  # 分位数输出层
        else:
            self.out_length_lay = nn.Linear(self.seq_len, self.output_seq_len)  # 普通输出层
        self.mask = generate_square_subsequent_mask(self.seq_len).to(device)  # 自回归掩码
        self.activation = nn.PReLU()  # 激活函数

    def make_embedding(self, x):
        '''
        创建模型输入的嵌入。
        '''
        x = self.dense_shape(x)  # 线性变换
        x = x.permute(1, 0, 2)  # 转置以适应Transformer的输入格式
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)  # 使用掩码
        else:
            x = self.transformer_enc(x)  # 不使用掩码
        return x

    def forward(self, x):
        '''
        前向传播方法。
        '''
        x = self.dense_shape(x)  # 线性变换
        x = x.permute(1, 0, 2)  # 转置
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)  # 使用掩码
        else:
            xiolk, m = self.transformer_enc(x)  # 不使用掩码
        x = self.output_dim_layer(x)  # 输出线性层
        x = x.permute(1, 2, 0)  # 转置
        x = self.activation(x)  # 激活函数
        x = self.out_length_lay(x)  # 输出层
        if self.output_dim > 1:
            return x.permute(0, 2, 1)
        if self.quantiles:
            return x.view(-1, len(self.quantiles))  # 分位数预测
        else:
            return x.view(-1, self.output_seq_len)  # 普通预测

def generate_square_subsequent_mask(sz):
    '''
    生成自回归掩码。
    '''
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 上三角矩阵
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 填充
    return mask
```

**说明：**
- **`HFformer`** 类定义了一个基于Transformer的时间序列预测模型，具有以下组件：
  - **输入层**：使用线性层将原始时间序列数据转换为模型的嵌入维度（`d_model`）。
  - **脉冲激活函数**：通过`pytorch_spiking.SpikingActivation`实现脉冲神经网络的激活函数，增强模型的非线性表达能力。
  - **Transformer编码器**：由多个`TransformerEncoderLayer`组成，每层包含多头自注意力机制和前馈神经网络，并使用脉冲激活函数。
  - **输出层**：
    - 普通预测：通过线性层将Transformer编码器的输出映射到目标维度。
    - 分位数预测：通过线性层输出多个分位数的预测值。
  - **掩码**：如果`use_mask=True`，则生成自回归掩码，确保模型在预测时只能利用之前的时间步信息，防止信息泄露。

- **`generate_square_subsequent_mask`** 函数用于生成Transformer中的自回归掩码，确保模型在预测当前时间步时只能利用之前的时间步信息。

---

### 7. **定义自定义数据集类**

```python
# %%
class TimeSeriesDataset(Dataset):
    '''
    将LOB（Limit Order Book，限价订单簿）数据转换为模型输入的数据集类。
    '''
    def __init__(self, x, y):
        self.x = x.astype(np.float32)  # 确保输入数据类型为float32
        self.y = y.astype(np.float32)  # 确保目标数据类型为float32
        
    def __len__(self):
        return len(self.x)  # 返回数据集的长度

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])  # 返回指定索引的数据对（输入，目标）
```

**说明：**
- **`TimeSeriesDataset`** 类继承自`torch.utils.data.Dataset`，用于将输入数据和目标数据封装为PyTorch的数据集：
  - **`__init__`**：初始化数据集，确保数据类型为`float32`，以提高计算效率。
  - **`__len__`** 和 **`__getitem__`** 方法分别返回数据集的长度和指定索引的数据对（输入，目标）。
  
- 通过这种方式，可以方便地使用`DataLoader`进行批次加载和多线程加速。

---

### 8. **定义自定义损失函数**

```python
# %%
def quantile_loss(y, y_pred, quantile):
    '''
    计算分位数损失。
    标准分位数损失，如同主要TFT（Temporal Fusion Transformer）论文中的“训练过程”部分定义。
    '''
    if quantile < 0 or quantile > 1:
        raise ValueError(
            '非法的分位数值={}！值应该在0和1之间。'.format(
                quantile))
    
    prediction_underflow = y - y_pred
    q_loss = quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow)) + (
        1. - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))
    
    return torch.sum(q_loss, axis=-1)

# %%
criterion_dict = {"MAE": nn.L1Loss, "MSE": nn.MSELoss, "QuantileLoss": quantile_loss}  # 定义损失函数字典
```

**说明：**
- **`quantile_loss`** 函数实现了分位数损失（Quantile Loss），用于模型进行分位数预测时的损失计算：
  - 分位数损失能够使模型在不同的分位数上有不同的预测结果，适用于不确定性建模。
  - 如果预测值低于真实值，损失与分位数成正比；如果预测值高于真实值，损失与（1 - 分位数）成正比。
  
- **`criterion_dict`** 字典定义了多种损失函数的映射，包括：
  - **MAE**（平均绝对误差）：`nn.L1Loss`
  - **MSE**（均方误差）：`nn.MSELoss`
  - **QuantileLoss**：自定义的分位数损失函数

---

### 9. **定义辅助函数**

#### a. **计算损失**

```python
def compute_loss(labels, output, src, criterion, quantiles):
    '''
    计算损失。
    '''
    if isinstance(output, torch.Tensor):
        if len(labels.shape) != len(output.shape):
            if len(labels.shape) > 1:
                if labels.shape[1] == output.shape[1]:
                    labels = labels.unsqueeze(2)
                else:
                    labels = labels.unsqueeze(0)
    loss = 0
    if quantiles:
        for idx, quantile in enumerate(quantiles):
            loss += criterion(output[:, idx], labels.float(), quantile)  # 计算每个分位数的损失并累加
    else:
        loss = criterion(output, labels.float())  # 计算普通损失
    return loss
```

**说明：**
- **`compute_loss`** 函数根据是否使用分位数预测，计算相应的损失：
  - 如果使用分位数预测，遍历每个分位数，计算并累加对应的分位数损失。
  - 否则，直接使用指定的损失函数（如MSE或MAE）计算总损失。
  
- 此函数确保标签和输出的形状一致，必要时通过`unsqueeze`扩展维度。

#### b. **训练步骤**

```python
def train_step(model, opt, criterion, data_loader, takes_target, device,
               num_targets=1, forward_params={}):
    '''
    执行模型的单步训练。遍历一个epoch的数据。
    '''
    i = 0
    running_loss = 0.0
    model.train()  # 将模型设置为训练模式
    for src, trg in data_loader:
        opt.zero_grad()  # 清零梯度
        if takes_target:
            forward_params["t"] = trg.to(device)  # 如果需要目标作为额外输入，添加到forward_params
        src = src.to(device)  # 将输入数据移动到设备（GPU或CPU）
        trg = trg.to(device)  # 将目标数据移动到设备
        
        output = model(src, **forward_params)  # 前向传播
        output = output.squeeze()  # 压缩维度
        if num_targets == 1:
            labels = trg
        elif num_targets > 1:
            labels = trg[:, :, 0:num_targets]
    
        loss = compute_loss(labels, output, src, criterion[0], quantiles=model.quantiles)  # 计算损失
        loss.backward()  # 反向传播
        opt.step()  # 更新参数
        running_loss += loss.item()  # 累加损失
        i += 1
    total_loss = running_loss
    return total_loss
```

**说明：**
- **`train_step`** 函数执行一个训练epoch的步骤：
  - 将模型设置为训练模式，启用Dropout等训练专用层。
  - 遍历训练数据加载器中的所有批次，执行前向传播、计算损失、反向传播和优化步骤。
  - 累加每个批次的损失，返回总损失。

#### c. **验证步骤**

```python
def validation(val_loader, model, criterion, device, num_targets=1):
    '''
    计算验证集的损失指标。
    '''
    crit_losses = dict.fromkeys(criterion, 0)  # 初始化每种损失的累计值
    model.eval()  # 将模型设置为评估模式
    labels = torch.Tensor(0).to(device)
    labels_all = torch.Tensor(0).to(device)
    output_all = torch.Tensor(0).to(device)
    with torch.no_grad():  # 禁用梯度计算
        for src, targ in val_loader:
            output = torch.Tensor(0).to(device)
            src = src if isinstance(src, list) else src.to(device)  # 将输入数据移动到设备
            targ = targ if isinstance(targ, list) else targ.to(device)  # 将目标数据移动到设备
            output = model(src.float())  # 前向传播
            output = output.squeeze()  # 压缩维度
            output_all = torch.cat((output_all, output))  # 收集所有输出
            if num_targets == 1:
                labels = targ
            elif num_targets > 1:
                labels = targ[:, :, 0:num_targets]
            for crit in criterion:
                loss = compute_loss(labels, output, src, crit, model.quantiles)  # 计算每种损失
                crit_losses[crit] += loss.item()  # 累加损失
            labels_all = torch.cat((labels_all, labels))  # 收集所有标签
    return list(crit_losses.values())[0], output_all, labels_all  # 返回第一个损失值、所有输出和所有标签
```

**说明：**
- **`validation`** 函数用于在验证集上评估模型的性能：
  - 将模型设置为评估模式，禁用Dropout等训练专用层。
  - 禁用梯度计算，节省内存和计算资源。
  - 遍历验证数据加载器中的所有批次，执行前向传播，计算并累加每种损失函数的损失。
  - 返回验证集的总损失、模型的所有输出和真实标签。

#### d. **预测步骤**

```python
def forecast(data_loader, model, criterion, forecast_horizon, device, num_targets=1):
    '''
    进行预测。
    '''
    crit_losses = dict.fromkeys(criterion, 0)  # 初始化每种损失的累计值
    model.eval()  # 将模型设置为评估模式
    output_decoder = torch.Tensor(0).to(device)
    labels = torch.Tensor(0).to(device)
    labels_all = torch.Tensor(0).to(device)
    counter = 0
    with torch.no_grad():  # 禁用梯度计算
        for src, targ in data_loader:
            if (counter % forecast_horizon) == 0:  # 按预测窗口间隔进行预测
                src = src if isinstance(src, list) else src.to(device)  # 将输入数据移动到设备
                targ = targ if isinstance(targ, list) else targ.to(device)  # 将目标数据移动到设备
                output = model(src.float())  # 前向传播
                output_decoder = torch.cat((output_decoder, output))  # 收集所有输出
                if num_targets == 1:
                    labels = targ
                elif num_targets > 1:
                    labels = targ[:, :, 0:num_targets]
                for crit in criterion:
                    loss = compute_loss(labels, output, src, crit, model.quantiles)  # 计算每种损失
                    crit_losses[crit] += loss.item()  # 累加损失
                labels_all = torch.cat((labels_all, labels))  # 收集所有标签
            counter += 1
    return list(crit_losses.values())[0], output_decoder, labels_all  # 返回第一个损失值、所有输出和所有标签
```

**说明：**
- **`forecast`** 函数用于在测试集上进行预测和评估：
  - 将模型设置为评估模式，禁用Dropout等训练专用层。
  - 按照预测窗口（`forecast_horizon`）的间隔进行预测，避免频繁预测，提高效率。
  - 累加每种损失函数的损失，并收集所有预测值和真实标签。
  - 返回损失值、所有预测输出和真实标签，用于后续评估。

---

### 10. **定义训练器函数**

```python
# %%
def strategy_evaluator(true, pred):
    '''
    评估交易策略的正确买卖和持有情况。
    '''
    total_buys, total_sells, total_holds = np.sum(true > 0), np.sum(true < 0), np.sum(true == 0)  # 总买入、卖出、持有次数
    total_correct_buys, total_correct_sells, total_correct_holds = 0, 0, 0  # 正确买入、卖出、持有次数
    for idx in range(len(true)):
        for jdx in range(len(true[0])):
            if true[idx, jdx] > 0 and pred[idx, jdx] > 0:
                total_correct_buys += 1  # 正确买入
            elif true[idx, jdx] < 0 and pred[idx, jdx] < 0:
                total_correct_sells += 1  # 正确卖出
            elif true[idx, jdx] == 0 and pred[idx, jdx] == 0:
                total_correct_holds += 1  # 正确持有
    # 计算正确率
    total_correct_buys_r = (total_correct_buys / total_buys) if total_buys > 0 else 0
    total_correct_sells_r = (total_correct_sells / total_sells) if total_sells > 0 else 0
    total_correct_holds_r = (total_correct_holds / total_holds) if total_holds > 0 else 0
    return total_correct_buys_r.round(3), total_correct_sells_r.round(3), total_correct_holds_r.round(3)

# %%
def trainer(model, train_loader, validation_loader, test_loader, criterion, opt, scheduler,
            warmup_scheduler, max_epochs, batch_size, forecast_horizon, takes_target, shuffle=False,
            num_targets=1, plot_prediction=True, save_path='/home/gaen/Documents/codespace-gaen/Simons/results_Transencwithlineardec',
            LAG=0):
    '''
    训练方法。
    '''
    start_time = time.time()
    
    # 创建数据加载器
    data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle, sampler=None, batch_sampler=None, num_workers=10)
    validation_data_loader = DataLoader(validation_loader, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=10)
    test_data_loader = DataLoader(test_loader, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=2)
    forecast_data_loader = DataLoader(validation_loader, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=2)
    
    for epoch in range(1, max_epochs + 1):  # 遍历每个epoch

        # 训练步骤
        total_loss = train_step(model, opt, criterion, data_loader, takes_target, device, num_targets=num_targets)
        val_loss = 0
        if plot_prediction:
            # 进行预测并绘图
            val_loss, val_values, true_values = forecast(forecast_data_loader, model, criterion, forecast_horizon=forecast_horizon,
                                                           device=device, num_targets=num_targets)
            fig, ax = plt.subplots(1, 1, figsize=(18, 8))
            ax.plot(true_values.cpu().view(-1), label='truth', alpha=0.3)
            ax.plot(val_values.cpu().view(-1), label='forecast', alpha=0.8)
            ax.set_xlim(left=0, right=len(true_values.cpu().view(-1)))
            plt.show()
        else:
            # 计算验证集的损失
            val_loss, val_values, true_values = validation(validation_data_loader, model, criterion, device,
                                                            num_targets=num_targets)
        
        preds, trues = val_values.cpu().numpy(), true_values.cpu().numpy()  # 获取预测值和真实值

        print(f'preds {preds.shape}')  # 打印预测值形状
        print(f'trues {trues.shape}')  # 打印真实值形状

        results = 0
        if model.quantiles:
            # 如果使用分位数预测，计算中位数对应的R²评分
            r2_sklearn = r2_score(trues, preds[:, len(model.quantiles) // 2])
        else:
            # 计算普通的R²评分
            r2_sklearn = r2_score(trues, preds)

        elapsed = time.time() - start_time  # 计算时间
        print('-' * 88)
        print('| epoch {:3d} | {:5.2f} s | train loss {:5.5f} | val loss {:5.5f} | lr {:1.8f} | r2 sklearn: {:1.5f} | b, s, h: {:}|'.format(
                        epoch, elapsed, total_loss, val_loss, scheduler.get_last_lr()[0], r2_sklearn, results))
        print('-' * 88)
        start_time = time.time()

        if save_path:
            results = {
                'model': 'Transencwithlineardec',
                'pred_len': forecast_horizon,
                'epoch': epoch,
                'train_loss': total_loss,
                'val_loss': val_loss,
                'r2_val_sklearn': r2_sklearn            
            }

            df = pd.DataFrame([results])
            df.to_csv(os.path.join(save_path, 'results.csv'), mode='a', header=not os.path.exists(save_path), index=False)

            # 如果R²评分超过阈值，则保存模型参数
            if r2_sklearn > 0.02:
                torch.save(model.state_dict(), os.path.join(save_path, "Transencwithlineardec", f'_epoch_{epoch}_time_{time.time()}_r2_{r2_sklearn}.pt'))

        with warmup_scheduler.dampening():  # 更新学习率调度器
            scheduler.step()
```

**说明：**
- **`strategy_evaluator`** 函数用于评估基于模型预测的交易策略的效果：
  - 统计总的买入、卖出和持有次数。
  - 计算预测与真实标签中正确的买入、卖出和持有的比例。
  - 返回这三个比例，反映模型在交易决策上的准确性。
  
- **`trainer`** 函数负责整个模型的训练过程：
  - **数据加载器**：将训练集、验证集、测试集封装为`DataLoader`，设置批次大小和多线程加载。
  - **训练循环**：遍历指定的`max_epochs`，每个epoch包括：
    - 执行一个训练步骤，计算训练损失。
    - 执行验证步骤，计算验证损失和R²评分。
    - 如果设置了`plot_prediction=True`，则绘制真实值与预测值的对比图。
    - 打印每个epoch的详细信息，包括损失、学习率和R²评分。
    - 将训练结果保存到CSV文件中，并根据R²评分保存模型参数。
    - 更新学习率调度器。

---

### 11. **模型训练与参数搜索**

```python
# %%
''' 
此代码块被注释掉，可能是之前的训练过程。以下部分未被注释，主要用于模型的参数搜索和训练。
'''
```

**说明：**
- 注释掉的代码块可能是之前的训练过程，接下来的代码主要用于在不同的预测窗口长度下训练模型，并保存训练结果和模型参数。

```python
# %%
date_train = 'all' 
date_test = 'all'
date_train = 'All_to_Sept'  # 设置训练数据的日期范围
date_test = 'All_to_Sept'  # 设置测试数据的日期范围

drive = None  # 标记是否使用Google Drive
if False:
    if drive:
        agg_trade = pd.read_csv(config["paths"]["drive"]["agg_trade"]["train"] + date_train + '/orderbook.csv')    
        sys.path.append(config["paths"]["drive"]["utils"])
    else: 
        agg_trade = pd.read_csv(config["paths"]["local"]["agg_trade"]["train"] + date_train + '/orderbook_agg_trade_dollarvol_drop_duplicate_price.csv')
        agg_trade_test = pd.read_csv(config["paths"]["local"]["agg_trade"]["test"] + date_test + '/orderbook_agg_trade_dollarvol_drop_duplicate_price.csv')
idx = 0
agg_trade = pd.read_csv(config["paths"]["local"]["agg_trade"]["train"] + date_train + '/orderbook_agg_trade_dollarvol.csv')  # 读取本地训练数据
agg_trade_test = pd.read_csv(config["paths"]["local"]["agg_trade"]["test"] + date_test + '/orderbook_agg_trade_dollarvol.csv')  # 读取本地测试数据
agg_trade['w_midprice'] = (agg_trade['ask1'] * agg_trade['askqty1'] + agg_trade['bid1'] * agg_trade['bidqty1']) / (agg_trade['askqty1'] + agg_trade['bidqty1'])  # 计算加权中间价格
```

**说明：**
- 设置训练和测试数据的日期范围为`All_to_Sept`。
- 读取本地的订单簿数据文件，计算加权中间价格（`w_midprice`），作为模型的价格特征。

---

```python
# %%
model_name = 'HFfMODELsept10test'

save_path = os.path.join(f'./home/gaen/Documents/codespace-gaen/Simons/{model_name}/training_details/HFTransformer/results_HFformer',
                            str(int(time.time())) + '_results.csv')

save_path = f'/home/gaen/Documents/codespace-gaen/Simons/{model_name}/training_details/HFTransformer/results_HFformer'
filepath = f'/home/gaen/Documents/codespace-gaen/Simons/{model_name}/training_details/HFTransformer/results_HFformer/HFformer'
# 创建目录（如果不存在）
os.makedirs(save_path, exist_ok=True)
os.makedirs(filepath, exist_ok=True)

forecast_history = 100  # 预测历史窗口大小
epochs = 16  # 训练周期数
batch_size = 256  # 批次大小（对于线性解码器使用64）

forecast_windows = [i for i in range(1, 31)]  # 定义要测试的预测窗口长度（1到30）

for forecast_window in forecast_windows:
    
    orderbook = augment_trade_data(agg_trade, lag=0, forecast_window=forecast_window)  # 对数据进行增强处理

    features = ['price', 'lag_return',
                'bid1', 'bidqty1', 'bid2', 'bidqty2', 'bid3', 'bidqty3', 'bid4', 'bidqty4', 'bid5', 'bidqty5',
                'bid6', 'bidqty6', 'bid7', 'bidqty7', 'bid8', 'bidqty8', 'bid9', 'bidqty9',
                'ask1', 'askqty1', 'ask2', 'askqty2', 'ask3', 'askqty3', 'ask4', 'askqty4', 'ask5', 'askqty5',
                'ask6', 'askqty6', 'ask7', 'askqty7', 'ask8', 'askqty8', 'ask9', 'askqty9']

    # 准备训练和验证数据
    split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
        np.array(orderbook[features][1_000_000:1_720_000]),  # 选择训练数据的特定范围
        np.array(agg_trade.datetime[2_005_000 - 500_000:2_006_000 - 500_000]),  # 对应的日期时间
        np.array(orderbook[features][60_000:60_600]),  # 测试数据的特定范围
        np.array(agg_trade.datetime[60_000:60_600]),  # 对应的日期时间
        config, lag=forecast_window, plot=False
    )

    # 创建训练集和验证集的数据加载器
    train_loader = TimeSeriesDataset(data_x_train, data_y_train)
    val_loader = TimeSeriesDataset(data_x_val, data_y_val)
    test_loader = None  # 暂时没有测试集

    # 实例化HFformer模型
    model_custom = HFformer(n_time_series=len(features), seq_len=forecast_history, output_seq_len=1, d_model=36,
                  n_heads=6, n_layers_encoder=2, dropout=0.3, output_dim=1, forward_dim=64, use_mask=True).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.MSELoss(reduction='sum')  # 均方误差损失
    optimizer = optim.AdamW(model_custom.parameters(), lr=0.1, amsgrad=True)  # AdamW优化器，学习率为0.1
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # 指数衰减学习率调度器
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=1000)  # 线性预热调度器，预热周期为1000

    # 调用trainer函数进行模型训练
    trainer(model_custom, train_loader, val_loader, test_loader, [criterion], optimizer, scheduler, warmup_scheduler, epochs, batch_size=batch_size,
            forecast_horizon=forecast_window, takes_target=False, plot_prediction=False, save_path=save_path, LAG=forecast_window)
    
    # 删除不再需要的变量，释放内存
    del data_x_train 
    del data_y_train
    del data_x_val
    del data_y_val

    # 保存训练好的模型参数
    torch.save(model_custom, f'./{model_name}/transformer_enclinear_forecasting_FINAL_horizon_{forecast_window}.pt')
    print(f'Done with prediction len {forecast_window}.')  # 打印完成信息
```

**说明：**
- **模型名称与保存路径**：
  - 定义了模型的名称`HFfMODELsept10test`。
  - 设置了结果和模型参数的保存路径，并确保这些目录存在。
  
- **参数搜索循环**：
  - 定义了预测历史窗口大小`forecast_history`、训练周期数`epochs`和批次大小`batch_size`。
  - 定义了`forecast_windows`，即要测试的不同预测窗口长度（1到30）。
  
  - **循环过程**：
    1. 对每个`forecast_window`，通过`augment_trade_data`函数增强数据。
    2. 选择相关特征列，包括价格、对数收益率以及多个买卖价位和数量。
    3. 调用`prepare_data`函数准备训练和验证数据。
    4. 创建训练集和验证集的数据加载器。
    5. 实例化`HFformer`模型，设置相应的参数。
    6. 定义损失函数（MSE）、优化器（AdamW）、学习率调度器（ExponentialLR）和预热调度器（LinearWarmup）。
    7. 调用`trainer`函数进行模型训练，遍历不同的预测窗口长度，保存训练结果和模型参数。
    8. 删除不再需要的变量，释放内存。
    9. 保存训练好的模型参数到指定路径。
    10. 打印当前预测窗口的训练完成信息。
    
- **数据管道工作流程**：
  1. **数据增强**：通过对数差分和收益率计算，生成额外的特征，增强模型对未来价格变化的理解。
  2. **特征选择**：选择包括价格、对数收益率以及多个买卖价位和数量在内的特征。
  3. **数据窗口化**：将时间序列数据转换为固定大小的输入窗口和对应的预测目标。
  4. **数据加载**：使用PyTorch的`DataLoader`高效加载和批处理数据，支持多线程加速。
  5. **模型训练与评估**：在不同的预测窗口长度下，训练模型并评估其在验证集上的性能，选择最优参数组合。
  
---

### 12. **定义预测评估函数**

```python
# %%
def strategy_evaluator(true, pred, weighted=False):
    '''
    基于正确的买卖操作评估交易策略。
    '''
    total_buys, total_sells, total_holds = np.sum(true > 0), np.sum(true < 0), np.sum(true == 0)  # 总买入、卖出、持有次数
    total_correct_buys, total_correct_sells, total_correct_holds = 0, 0, 0  # 正确买入、卖出、持有次数
    for idx in range(len(true)): 
        if true[idx] > 0 and pred[idx] > 0:
            total_correct_buys += 1  # 正确买入
        elif true[idx] < 0 and pred[idx] < 0:
            total_correct_sells += 1  # 正确卖出
        elif true[idx] == 0 and pred[idx] == 0:
            total_correct_holds += 1  # 正确持有
    # 计算正确率
    total_correct_buys_r = (total_correct_buys / total_buys) if total_buys > 0 else 0
    total_correct_sells_r = (total_correct_sells / total_sells) if total_sells > 0 else 0
    total_correct_holds_r = (total_correct_holds / total_holds) if total_holds > 0 else 0
    total_correct_trades = (total_correct_buys + total_correct_sells + total_correct_holds) / (total_buys + total_sells + total_holds) if (total_buys + total_sells + total_holds) > 0 else 0
    return total_buys, total_correct_buys, total_sells, total_correct_sells, total_holds, total_correct_holds

def forecast_evaluator(test_loader, model, criterion, forecast_horizon=1, device=device, num_targets=1, save_path=None):
    '''
    输出评估指标。
    '''
    test_data_loader = DataLoader(test_loader, batch_size=128, shuffle=False, sampler=None, batch_sampler=None, num_workers=6)
    loss, pred, true = forecast(test_data_loader, model, criterion, forecast_horizon=1, device=device, num_targets=1)  # 进行预测
    pred, true = pred.cpu().numpy(), true.cpu().numpy()  # 转换为numpy数组

    r2 = r2_score(true, pred)  # 计算R²评分
    strategy_results = strategy_evaluator(true, pred)  # 评估交易策略

    if save_path:
        results = {
            'model': 'HFformer',
            'pred_len': forecast_horizon,
            'test_loss': loss,
            'r2_val_sklearn': r2,
            'correct_buys': strategy_results[1],
            'total_buys':  strategy_results[0],
            'correct_sells': strategy_results[3],
            'total_sells': strategy_results[2],
            'correct_holds': strategy_results[5],
            'total_holds': strategy_results[4],
        }

        df = pd.DataFrame([results])
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)  # 保存结果到CSV

    print(f'| test loss {loss} | b, cb, s, cs, h, ch: {strategy_results} |')  # 打印评估结果

    return pred, true
```

**说明：**
- **`strategy_evaluator`** 函数用于评估基于模型预测的交易策略的效果：
  - 统计总的买入、卖出和持有次数。
  - 计算预测与真实标签中正确的买入、卖出和持有的比例。
  - 返回这三个比例，以及总的买入、卖出和持有次数。
  
- **`forecast_evaluator`** 函数用于在测试集上进行预测，并计算相关评估指标：
  - 使用`forecast`函数进行预测，获取模型的预测值和真实值。
  - 计算R²评分，评估模型的拟合程度。
  - 调用`strategy_evaluator`函数评估交易策略的有效性。
  - 如果指定了`save_path`，则将评估结果保存到CSV文件中。
  - 打印测试集上的损失和交易策略评估结果。

---

### 13. **模型预测**

```python
# %%
date_train = 'all' 
date_test = 'all'
drive  = None  # 标记是否使用Google Drive
if drive:
    agg_trade = pd.read_csv(config["paths"]["drive"]["agg_trade"]["train"] + date_train + '/orderbook.csv')    
    sys.path.append(config["paths"]["drive"]["utils"])
else:
    agg_trade = pd.read_csv(config["paths"]["local"]["agg_trade"]["train"] + date_train + '/orderbook_agg_trade_dollarvol.csv')  # 读取训练数据
    agg_trade_test = pd.read_csv(config["paths"]["local"]["agg_trade"]["test"] + date_test + '/orderbook_agg_trade_dollarvol.csv')  # 读取测试数据

agg_trade_test = agg_trade[2_000_000:]  # 截取测试数据的特定范围
```

**说明：**
- 再次加载训练和测试数据，并对测试数据进行截取，选择特定的数据范围进行预测。
- 根据是否使用Google Drive，选择不同的数据路径进行读取。

---

```python
# %%
print(agg_trade_test.shape)  # 打印测试数据的形状，确认数据加载是否正确
```

**说明：**
- 打印测试数据的形状，确保数据加载成功且格式正确。

---

```python
# %%
save_path = os.path.join('/home/gaen/Documents/codespace-gaen/Simons/models/training_details/HFTransformer/results_HFformer',
                            str(int(time.time())) + '_forecasting_results.csv')  # 设置预测结果保存的CSV路径

# save_path = None  # 如果不需要保存，可以将save_path设置为None

save_path_results = os.path.join('/home/gaen/Documents/codespace-gaen/Simons/models/training_details/HFTransformer/results_HFformer',
                            str(int(time.time())) + '_list_results.pkl')  # 设置预测结果保存的Pickle路径

save_path_model = os.path.join('/home/gaen/Documents/codespace-gaen/Simons/models/training_details/HFTransformer/results_HFformer',
                                str(int(time.time())) + '_model.pth')  # 设置模型参数保存路径

forecast_history = 100  # 预测历史窗口大小
batch_size = 256  # 批次大小

forecast_windows = [i for i in range(1, 31)]  # 定义要测试的预测窗口长度（1到30）

predictions = []  # 存储所有预测结果

for forecast_window in forecast_windows:
    
    orderbook = augment_trade_data(agg_trade_test, lag=0, forecast_window=forecast_window)  # 对测试数据进行增强处理

    features = ['price', 'lag_return',
                'bid1', 'bidqty1', 'bid2', 'bidqty2', 'bid3', 'bidqty3', 'bid4', 'bidqty4', 'bid5', 'bidqty5',
                'bid6', 'bidqty6', 'bid7', 'bidqty7', 'bid8', 'bidqty8', 'bid9', 'bidqty9',
                'ask1', 'askqty1', 'ask2', 'askqty2', 'ask3', 'askqty3', 'ask4', 'askqty4', 'ask5', 'askqty5',
                'ask6', 'askqty6', 'ask7', 'askqty7', 'ask8', 'askqty8', 'ask9', 'askqty9']
    print(orderbook[features].shape)  # 打印选择的特征列的形状
    
    # 准备测试数据
    split_index, data_x_train, data_y_train, data_x_val, data_y_val = prepare_data(
        np.array(orderbook[features][:]),  # 使用全部数据作为输入
        np.array(agg_trade.datetime[2_005_000 - 500_000:2_006_00 - 5_000_000]),  # 对应的日期时间（注意这里可能有拼写错误，应为2_006_000 - 500_000）
        np.array(orderbook[features][60_000:60_600]),  # 测试数据的特定范围
        np.array(agg_trade.datetime[60_000:60_600]),  # 对应的日期时间
        config, lag=forecast_window, plot=True  # 设置滞后窗口和是否绘图
    )

    test_loader = TimeSeriesDataset(data_x_train, data_y_train)  # 创建测试集的数据加载器

    # 加载训练好的模型
    model = torch.load(f'/home/gaen/Documents/codespace-gaen/Simons/models/transformer_enclinear_forecasting_FINAL_horizon_{forecast_window}.pt')
    criterion = nn.MSELoss(reduction='sum')  # 定义损失函数（MSE）

    # 进行预测和评估
    pred, true = forecast_evaluator(test_loader, model, [criterion], forecast_horizon=1, device=device, num_targets=1, save_path=save_path)
    
    predictions.append((pred, true))  # 将预测结果添加到列表中

    print(f'Done with prediction len {forecast_window}.')  # 打印完成信息

# 将所有预测结果保存到Pickle文件中
with open(save_path_results, 'wb') as f:
    pickle.dump(predictions, f)
```

**说明：**
- **设置保存路径**：
  - `save_path`：用于保存预测结果的CSV文件路径。
  - `save_path_results`：用于保存预测结果的Pickle文件路径，方便后续分析。
  - `save_path_model`：用于保存模型参数的路径（当前代码未使用）。
  
- **预测循环**：
  - 遍历不同的预测窗口长度（1到30）。
  - 对每个`forecast_window`：
    1. 对测试数据进行增强处理，生成新的特征。
    2. 选择相关特征列，包括价格、对数收益率以及多个买卖价位和数量。
    3. 调用`prepare_data`函数准备测试数据。
    4. 创建测试集的数据加载器。
    5. 加载对应预测窗口的训练好的模型参数。
    6. 定义损失函数（MSE）。
    7. 调用`forecast_evaluator`函数进行预测和评估，保存结果到CSV文件。
    8. 将预测结果（预测值和真实值）存储到`predictions`列表中。
    9. 打印当前预测窗口的预测完成信息。
    
- **保存预测结果**：
  - 使用`pickle`将所有预测结果保存到指定的Pickle文件中，便于后续分析和使用。

---

### 14. **总结**

整个代码的核心工作流程如下：

1. **数据加载与增强**：
   - 从本地或指定路径加载订单簿数据。
   - 通过`augment_trade_data`函数生成对数收益率等特征，模拟交易延迟。

2. **数据准备与窗口化**：
   - 选择特定的特征列（价格、买卖价位及数量等）。
   - 使用`prepare_data`函数将时间序列数据窗口化，生成模型的输入和目标。
   - 将数据划分为训练集和验证集，供模型训练和评估使用。

3. **模型定义与训练**：
   - 定义基于Transformer的`HFformer`模型，包含多头自注意力机制和脉冲激活函数。
   - 设置损失函数、优化器、学习率调度器和预热调度器。
   - 使用`trainer`函数进行模型训练，遍历不同的预测窗口长度，保存训练结果和模型参数。

4. **模型评估与预测**：
   - 定义`strategy_evaluator`和`forecast_evaluator`函数，用于评估模型在交易策略上的表现。
   - 遍历不同的预测窗口长度，加载对应的训练好的模型，进行预测和评估。
   - 保存预测结果和评估指标，便于后续分析。

**数据管道的关键步骤**：

- **数据增强**：通过对数差分和收益率计算，生成额外的特征，增强模型对未来价格变化的理解。
- **窗口化处理**：将连续的时间序列数据分割成固定长度的窗口，形成模型的输入序列和对应的预测目标。
- **标准化**：对输入数据进行标准化处理，减去均值除以标准差，提高模型训练的稳定性和收敛速度。
- **数据加载**：使用PyTorch的`DataLoader`高效加载和批处理数据，支持多线程加速。
- **模型训练与评估**：在不同的预测窗口长度下，训练模型并评估其在验证集和测试集上的性能，选择最优参数组合。

通过上述模块和步骤，整个数据管道实现了从原始订单簿数据到模型预测结果的完整流程，为金融时间序列预测任务提供了系统化的解决方案。