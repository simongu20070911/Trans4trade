以下是对您提供的 Jupyter Notebook (.ipynb) 中各个代码块的逐块描述和注释：

代码块 1

# %%
!pip --quiet install pytorch_spiking

描述：

	•	功能：使用 pip 安装 pytorch_spiking 库，--quiet 参数用于减少安装过程中的输出信息。
	•	注释：pytorch_spiking 是一个用于脉冲神经网络（Spiking Neural Networks）的 PyTorch 扩展库，可能用于实现更高效或更生物逼真的神经网络模型。

代码块 2

# %%
import os, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torch import nn
import pytorch_spiking
import matplotlib.pyplot as plt
import random
import pickle

描述：

	•	功能：导入所需的 Python 库和模块。
	•	导入内容：
	•	os：用于操作系统相关的功能，如文件路径管理。
	•	torch 及其子模块：用于构建和训练神经网络。
	•	pandas 和 numpy：用于数据处理和数值计算。
	•	pytorch_spiking：导入之前安装的脉冲神经网络扩展库。
	•	matplotlib.pyplot：用于绘图和可视化。
	•	random 和 pickle：用于随机操作和对象序列化。

代码块 3

# %%
#from google.colab import drive
#drive.mount('/home/gaen/Documents/codespace-gaen/Simons/backtest/{load_model_name}/drive')

描述：

	•	功能：这部分代码被注释掉了，原本用于在 Google Colab 中挂载 Google Drive。
	•	注释：如果在 Google Colab 环境中运行，可以取消注释以访问存储在 Google Drive 上的文件。

代码块 4

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

load_model_name = 'HFfMODELsept10test'
%cd /home/gaen/Documents/codespace-gaen/Simons/backtest
load_orderbook_name = 'All_to_Sept'
model_file_name_struct = 'transformer_enclinear_forecasting_FINAL_horizon_'
save_path = f'/home/gaen/Documents/codespace-gaen/Simons/backtest/{load_model_name}'
os.makedirs(save_path, exist_ok=True)
#os.makedirs(filepath, exist_ok=True)

#Note: if running cuda need to modify pytorch_spiking function's source code directly

描述：

	•	功能：
	•	设备设置：检测是否有可用的 CUDA 设备（GPU），若有则使用 GPU，否则使用 CPU。
	•	路径和文件名配置：
	•	设置模型名称 load_model_name。
	•	使用 %cd 魔法命令切换当前工作目录到指定路径。
	•	设置订单簿数据名称 load_orderbook_name。
	•	定义模型文件名结构 model_file_name_struct。
	•	设置保存路径 save_path 并确保该路径存在（若不存在则创建）。
	•	注释：
	•	提醒如果在 CUDA 上运行，可能需要直接修改 pytorch_spiking 函数的源代码。

代码块 5：Markdown

# %% [markdown]
## Models

描述：

	•	功能：这是一个 Markdown 单元，用于在 Notebook 中添加标题“Models”（模型）。
	•	注释：用于组织和说明后续代码块中定义的模型相关内容。

代码块 6

# %%
class TimeSeriesDataset(Dataset):
    '''
    Class for converting FTS data into time series for the ML models.
    '''
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def prepare_data_x(data, window_size, lag):
    '''
    Windows the input data for the ML models.
    '''
    subset = data[:window_size]
    subset_mean = np.mean(subset, axis=0)
    output = np.zeros([1, window_size, len(subset_mean)])
    x_mean = np.zeros([1, len(subset_mean)])
    x_std = np.zeros([1, len(subset_mean)])
    for idx in range(1):
        subset = data[idx:idx+window_size]
        subset_mean = np.mean(subset, axis=0)
        subset_std = np.std(subset, axis=0) + 0.01
        subset_norm = (subset-subset_mean)/subset_std
        x_mean[idx,:] = subset_mean
        x_std[idx,:] = subset_std
        output[idx,:,:] = subset_norm
    return output

def prepare_data_y(x, window_size, lag, delay):
    '''
    Windows the target data for the ML models.
    '''
    output = np.zeros([1,1,1])
    x = x.values
    std = 1.1*np.sqrt(lag)+lag*0.01
    future_price = x[window_size+lag-1,0]
    future_ask = x[window_size+lag-1,20]
    future_bid = x[window_size+lag-1,2]
    current_price = x[window_size+delay-1, 0]
    delayed_ask = x[window_size+delay-1, 20]
    delayed_bid = x[window_size+delay-1, 2]
    logreturn = np.log(future_price/current_price)*10_000
    output[0,0,0] = logreturn
    return output, current_price, future_price, delayed_bid, delayed_ask, future_bid, future_ask

def prepare_data(normalized_prices_train, delay, lag=1):
    '''
    Returns input and target data.
    '''
    data_x = prepare_data_x(normalized_prices_train, window_size=100, lag=lag)
    data_y, current_price, future_price, delayed_bid, delayed_ask, future_bid, future_ask = prepare_data_y(normalized_prices_train, window_size=100, lag=lag, delay=delay)
    return data_x, data_y, current_price, future_price, delayed_bid, delayed_ask, future_bid, future_ask

描述：

	•	功能：
	•	定义 TimeSeriesDataset 类：
	•	继承自 torch.utils.data.Dataset，用于将时间序列数据转换为适合机器学习模型训练的格式。
	•	__init__ 方法：初始化输入 x 和目标 y，并将其转换为 float32 类型。
	•	__len__ 方法：返回数据集的长度。
	•	__getitem__ 方法：根据索引返回对应的输入和目标对。
	•	定义数据准备函数：
	•	prepare_data_x：对输入数据进行窗口化处理，计算均值和标准差，并进行归一化。
	•	prepare_data_y：对目标数据进行窗口化处理，计算未来价格的对数收益率作为预测目标，并返回相关价格信息。
	•	prepare_data：整合 prepare_data_x 和 prepare_data_y，返回模型训练所需的输入和目标数据以及相关价格信息。
	•	注释：
	•	这些函数和类用于将原始订单簿数据转换为适合模型输入的格式，方便后续的训练和预测。

代码块 7

# %%
class HFformer(nn.Module):
    '''
    The HFformer model.
    '''
    def __init__(self, n_time_series, seq_len, output_seq_len, d_model=128, n_heads=8,
                 n_layers_encoder=6, dropout=0.1, output_dim=1, forward_dim=2048, use_mask=False):
        super(HFformer, self).__init__()
        self.n_time_series = n_time_series
        self.d_model = d_model
        self.nheads = n_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.n_layers_encoder = n_layers_encoder
        self.output_dim = output_dim 
        self.seq_len = seq_len
        self.output_seq_len = output_seq_len
        self.mask_it = use_mask

        self.dense_shape = nn.Linear(self.n_time_series, self.d_model)
        self.encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads, self.forward_dim, self.dropout, activation=pytorch_spiking.SpikingActivation(nn.PReLU()))
        self.encoder_norm = LayerNorm(self.d_model)
        self.transformer_enc = TransformerEncoder(self.encoder_layer, self.n_layers_encoder, self.encoder_norm)
        self.output_dim_layer = nn.Linear(self.d_model, self.output_dim)
        self.out_length_lay = nn.Linear(self.seq_len, self.output_seq_len)
        self.mask = generate_square_subsequent_mask(self.seq_len).to(device)
        self.activation = nn.PReLU()
    
    def make_embedding(self, x):
        x = self.dense_shape(x)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            x = self.transformer_enc(x)
        return x

    def forward(self, x):
        x = self.dense_shape(x)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            x = self.transformer_enc(x)
        x = self.output_dim_layer(x)
        x = x.permute(1, 2, 0)
        x = self.activation(x)
        x = self.out_length_lay(x)
        if self.output_dim > 1:
            return x.permute(0, 2, 1)
        return x.view(-1, self.output_seq_len)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

描述：

	•	功能：
	•	定义 HFformer 类：
	•	继承自 torch.nn.Module，实现了一个基于 Transformer 编码器的模型，结合了脉冲激活函数 (SpikingActivation)。
	•	初始化参数：
	•	n_time_series：时间序列的数量。
	•	seq_len：输入序列长度。
	•	output_seq_len：输出序列长度。
	•	其他参数如模型维度、头数、层数、dropout 比例等。
	•	组件：
	•	dense_shape：线性层，将输入时间序列映射到模型维度。
	•	encoder_layer：Transformer 编码器层，使用脉冲激活函数。
	•	transformer_enc：由多个编码器层组成的 Transformer 编码器。
	•	output_dim_layer 和 out_length_lay：用于生成最终输出。
	•	mask：生成的后续掩码，用于控制序列中位置之间的依赖关系。
	•	activation：PReLU 激活函数。
	•	方法：
	•	make_embedding：处理输入数据的嵌入。
	•	forward：定义前向传播过程，处理输入数据并生成预测输出。
	•	定义 generate_square_subsequent_mask 函数：
	•	生成一个上三角矩阵的掩码，用于 Transformer 中的自回归模型，防止模型看到未来的信息。
	•	注释：
	•	该模型结合了 Transformer 的强大序列建模能力和脉冲神经网络的特性，适用于复杂的时间序列预测任务。

代码块 8

# %%
class LSTM_MO(nn.Module):
    '''
    The many-to-one LSTM model.
    '''
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.prelu = nn.PReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.linear_1(x)
        x = self.prelu(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

描述：

	•	功能：
	•	定义 LSTM_MO 类：
	•	继承自 torch.nn.Module，实现了一个多层的 LSTM 模型，用于多对一的时间序列预测。
	•	初始化参数：
	•	input_size：输入特征的数量（默认为1）。
	•	hidden_layer_size：LSTM 隐藏层的大小。
	•	num_layers：LSTM 的层数。
	•	output_size：输出特征的数量（默认为1）。
	•	dropout：Dropout 比例，用于防止过拟合。
	•	组件：
	•	linear_1：线性层，将输入映射到隐藏层大小。
	•	prelu：PReLU 激活函数。
	•	lstm：多层 LSTM 模型。
	•	dropout：Dropout 层。
	•	linear_2：线性层，将 LSTM 的输出映射到最终输出。
	•	方法：
	•	init_weights：初始化 LSTM 的权重和偏置，使用不同的初始化方法提高模型的训练效果。
	•	forward：定义前向传播过程，处理输入数据并生成预测输出。
	•	注释：
	•	该 LSTM 模型用于处理时间序列数据，通过多层 LSTM 能够捕捉复杂的时间依赖关系，适合许多预测任务。

代码块 9：Markdown

# %% [markdown]
## Loading Data

描述：

	•	功能：这是一个 Markdown 单元，用于在 Notebook 中添加标题“Loading Data”（加载数据）。
	•	注释：用于组织和说明后续代码块中与数据加载相关的内容。

代码块 10

# %%
def augment_trade_data(df, lag, forecast_window=None):
    '''
    Augmenting input data.
    '''
    if forecast_window:
        df['lag_return'] = np.log(df['price'].shift(forecast_window)/df['price'].shift(forecast_window+1))
        return df.iloc[forecast_window+1:,:]
    if lag == 0:
        return df
    else:
        col_name = 'log_lag'+str(lag)+'_price'
        df[col_name] = np.log(df.price) - np.log(df.price).shift(lag)
        return df.iloc[lag:,:]

描述：

	•	功能：定义一个数据增强函数 augment_trade_data，用于扩展和处理交易数据。
	•	功能细节：
	•	参数：
	•	df：输入的 DataFrame，包含交易数据。
	•	lag：滞后步数，用于计算滞后收益。
	•	forecast_window：预测窗口大小，用于计算未来收益。
	•	逻辑：
	•	如果 forecast_window 被指定，计算未来 forecast_window 步的对数收益率，并返回处理后的数据。
	•	如果 lag 为 0，直接返回原始数据。
	•	否则，计算当前价格与 lag 步前价格的对数差值，作为新的特征，并返回相应的数据。
	•	注释：
	•	该函数用于生成额外的特征，增强模型的输入信息，如滞后收益和未来收益，以提高预测性能。

代码块 11

# %%
orderbook = pd.read_csv(f'/home/gaen/Documents/codespace-gaen/Simons/input_data/{load_orderbook_name}/orderbook_agg_trade_dollarvol.csv')
#orderbook['price'] = orderbook['w_midprice']

描述：

	•	功能：
	•	从指定路径读取订单簿数据的 CSV 文件，并加载到 Pandas DataFrame 中。
	•	注释掉的代码行用于替换 price 列为加权中间价 w_midprice，可能用于不同的价格计算方法。
	•	注释：
	•	确保路径和文件名正确，数据文件应包含必要的订单簿和交易量信息。

代码块 12

# %%
orderbook.shape
plt.plot(orderbook['price'])
orderbook_used = orderbook.iloc[2000000:3500000,:]
print(orderbook_used['datetime'])
plt.plot(orderbook_used['price'])

描述：

	•	功能：
	•	打印 orderbook 数据的形状（行数和列数）。
	•	绘制整个订单簿数据的价格走势。
	•	选择订单簿数据的一个子集（从第 2,000,000 行到第 3,500,000 行）。
	•	打印子集的 datetime 列（假设存在该列）。
	•	绘制子集数据的价格走势。
	•	注释：
	•	通过可视化和数据切片，检查数据的质量和特征分布，确保选取的数据段适合后续分析和建模。

代码块 13：Markdown

# %% [markdown]
## One run trades with one and multiple signals

描述：

	•	功能：这是一个 Markdown 单元，用于在 Notebook 中添加标题“One run trades with one and multiple signals”（一次交易运行，使用一个或多个信号）。
	•	注释：用于组织和说明后续代码块中涉及基于单个或多个信号的交易策略实现。

代码块 14

# %%
fee = 0.000002 # fee to simulate price slippage, set to 0.002%
fee = 0.0000002 # fee to simulate price slippage, set to 0.002%

#fee = 0.0001
delay = 2 # delay for placing trade
quantity = 1 # quantity traded
horizon = 25 # main forecasting horizon
input_window = 100 # size of the input window
hfformer = True # True if using HFformer, False if using LSTM
size_trades = True # use size trading 
num_models = 7 # number of models to use to generate trading signal 
min_threshold = True # True if ignoring trade signals below a certain threshold

描述：

	•	功能：设置交易策略的参数。
	•	参数说明：
	•	fee：交易费用，用于模拟价格滑点。两次赋值后最终 fee = 0.0000002（可能为错误，应检查）。
	•	delay：下单延迟，单位不明确，可能为步数或时间单位。
	•	quantity：每次交易的数量。
	•	horizon：主要预测的时间窗口长度。
	•	input_window：输入窗口的大小，用于模型的输入数据。
	•	hfformer：布尔值，决定使用 HFformer 模型还是 LSTM 模型。
	•	size_trades：布尔值，决定是否根据信号大小调整交易量。
	•	num_models：用于生成交易信号的模型数量。
	•	min_threshold：布尔值，决定是否忽略低于某个阈值的交易信号。
	•	注释：
	•	注意到 fee 被赋值两次，且第二次赋值的数值非常小，可能是误操作，应确认实际需要的交易费用比例。

代码块 15

# %%
orderbook_subset = augment_trade_data(orderbook_used, lag=0, forecast_window=horizon)

features = ['price', 'lag_return',
                'bid1', 'bidqty1', 'bid2', 'bidqty2', 'bid3', 'bidqty3', 'bid4', 'bidqty4', 'bid5', 'bidqty5',
                'bid6', 'bidqty6', 'bid7', 'bidqty7', 'bid8', 'bidqty8', 'bid9', 'bidqty9',# 'bid10', 'bidqty10',
                'ask1', 'askqty1', 'ask2', 'askqty2', 'ask3', 'askqty3', 'ask4', 'askqty4', 'ask5', 'askqty5',
                'ask6', 'askqty6', 'ask7', 'askqty7', 'ask8', 'askqty8', 'ask9', 'askqty9']#, 'ask10', 'askqty10']

orderbook_subset = orderbook_subset[features]

描述：

	•	功能：
	•	使用之前定义的 augment_trade_data 函数对订单簿数据进行增强，设置 lag=0 和 forecast_window=horizon。
	•	定义需要使用的特征列表 features，包括价格、滞后收益、各级别的买卖盘价格和数量。
	•	从增强后的数据集中选择这些特征列，形成最终用于建模的数据子集 orderbook_subset。
	•	注释：
	•	选择了订单簿中前 9 级的买卖盘信息（bid1 至 bid9 和 ask1 至 ask9），并注释掉了第 10 级，可能是为了简化模型或数据量。
	•	这些特征能够提供丰富的市场深度信息，有助于模型捕捉价格变动的微观结构。

代码块 16

# %%
models = []
ids = [id for id in range(horizon-num_models//2, horizon+num_models//2+1)]
ids = [23,24,25,26,28]

for idx in ids:
    if hfformer:
        models.append(torch.load(f'/home/gaen/Documents/codespace-gaen/Simons/{load_model_name}/{model_file_name_struct}{idx}.pt'))
    else:
        models.append(torch.load(f'/home/gaen/Documents/codespace-gaen/Simons/{load_model_name}/LSTM_MO_FINAL_LAG_{idx}.pt'))

描述：

	•	功能：
	•	初始化一个空列表 models，用于存储加载的模型。
	•	生成模型的 ID 列表 ids，原本基于 horizon 和 num_models 计算范围，但随后被手动指定为 [23,24,25,26,28]。
	•	根据 hfformer 参数，决定加载 HFformer 模型或 LSTM 模型，将其加载到 models 列表中。
	•	注释：
	•	torch.load 用于加载之前训练好的模型文件。
	•	模型文件名结构根据 horizon 和 ids 动态生成，确保加载正确的预测模型。
	•	注意 ids 列表被手动指定为 [23,24,25,26,28]，可能是基于特定的实验设计或模型选择。

代码块 17

# %%
notional = 40_000
holdings = 0
max_holdings = 0.1
min_holdings = -0.1
pnl = 0
pnls = []
cum_pnls = []
cum_notionals = []
total_fees = 0
trade_sizes = [0.15, 0.125, 0.1, 0.075, 0.05, 0.025] #[0.15, 0.125, 0.1, 0.075, 0.05, 0.025] #[0.15, 0.1, 0.05]
trade_size_thresholds = [0.15, 0.125, 0.1, 0.075, 0.05] #[0.25, 0.2, 0.15, 0.1, 0.05] #[0.15, 0.05]
quantities = []

verbose = True

signals_start = []
signals_end = []
signals = []

prev_price = 0

prev_ask = 0
prev_bid = 0
lob_prev_price = 0
start = 0
end = 800_000

描述：

	•	功能：初始化交易策略的相关变量和参数。
	•	变量说明：
	•	notional：初始资金量，设为 40,000。
	•	holdings：当前持仓量，初始为 0。
	•	max_holdings 和 min_holdings：持仓量的上下限，分别为 0.1 和 -0.1。
	•	pnl：总盈亏，初始为 0。
	•	pnls：记录每笔交易的盈亏。
	•	cum_pnls：累计盈亏。
	•	cum_notionals：累计资金量。
	•	total_fees：累计交易费用。
	•	trade_sizes：不同交易信号强度对应的交易量。
	•	trade_size_thresholds：对应的交易信号阈值。
	•	quantities：记录每笔交易的数量。
	•	verbose：控制是否打印详细的交易信息。
	•	signals_start 和 signals_end：记录交易开始和结束时的信号。
	•	signals：记录所有交易信号。
	•	prev_price、prev_ask、prev_bid、lob_prev_price：记录前一时刻的价格信息。
	•	start 和 end：定义回测的时间范围，从 0 到 800,000 步。
	•	注释：
	•	这些变量和参数用于跟踪和记录交易过程中的各种状态和指标，帮助评估策略的表现。

代码块 18

# %%
for timestep in range(start + input_window, end, 1):
    if timestep % horizon == 0 and timestep+horizon < len(orderbook_subset):

        input_data = orderbook_subset[timestep-input_window:timestep+horizon].copy()
        hist_lob = input_data.values
        data_x, data_y, current_midprice, future_midprice, current_bid_price, current_ask_price, future_bid, future_ask = prepare_data(input_data, lag=horizon, delay=delay)
        ts_dataset = TimeSeriesDataset(data_x, data_y)
        test_loader = DataLoader(ts_dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)

            if hfformer:
                signals.append(np.array([model(src.float()).detach().cpu().numpy()[0][0] for model in models]))
            else:
                signals.append(np.array([model(src.float()).detach().cpu().numpy()[0] for model in models]))

            trg = trg.detach().cpu().numpy()[0][0][0]

            if size_trades and holdings == 0:
                quantity = 0
                for idx, threshold in enumerate(trade_size_thresholds):
                    if np.abs(np.sum(signals[-1])) > threshold*len(signals[-1]):
                        quantity = trade_sizes[idx]
                        break
                if not quantity:  
                    quantity = trade_sizes[-1]
            
            if min_threshold and np.abs(np.sum(signals[-1])) < len(signals[-1])*0.1 and holdings == 0:
                print(f'\n Not trading as trade signal below threshold \n')
                continue

            if np.all((signals[-1] > 0)) and holdings == 0:
                holdings += quantity
                notional -= current_midprice*quantity
                notional -= current_midprice*quantity*fee
                prev_ask = current_midprice
                signals_start.append(signals[-1][len(signals[-1])//2])
                quantities.append(quantity)
                if verbose:
                    print(f'| Long     | Quantity {quantity} | Ask Price {current_midprice:.2f} | Holdings {holdings} |')
                    # print(f'future midprice {current_midprice:.2f}')
                continue
            elif holdings > 0:
                holdings -= quantity
                notional += current_midprice*quantity 
                notional -= current_midprice*quantity*fee
                prev_bid = current_midprice
                trade_pnl = (current_midprice - prev_ask)*quantity - (prev_ask+current_midprice)*quantity*fee
                pnl += trade_pnl
                pnls.append(trade_pnl)
                cum_pnls.append(pnl)
                signals_end.append(signals[-1][len(signals[-1])//2])
                total_fees += (prev_ask+current_midprice)*quantity*fee
                if verbose:
                    print(f'| Short    | Quantity {quantity} | Bid price {current_midprice:.2f} |')
                    print(f'| PnL {(current_midprice - prev_ask)*quantity:.2f} | Holdings {holdings} |')
                    print('\n')
                continue
            

            if np.all((signals[-1] < 0)) and holdings == 0:
                holdings -= quantity
                notional += current_midprice*quantity
                notional -= current_midprice*quantity*fee
                prev_bid = current_midprice
                signals_start.append(signals[-1][len(signals[-1])//2])
                quantities.append(quantity)
                if verbose:
                    print(f'| Short    | Quantity {quantity} | Bid Price {current_midprice:.2f} | Holdings {holdings} |')
                    # print(f'future midprice {current_midprice:.2f}')
                continue
            elif holdings < 0:
                holdings += quantity
                notional -= current_midprice*quantity 
                notional -= current_midprice*quantity*fee
                prev_ask = current_midprice
                trade_pnl = (prev_bid - current_midprice)*quantity - (prev_bid+current_midprice)*quantity*fee
                pnl += trade_pnl
                pnls.append(trade_pnl)
                cum_pnls.append(pnl)
                signals_end.append(signals[-1][len(signals[-1])//2])
                total_fees += (prev_bid+current_midprice)*quantity*fee
                if verbose:
                    print(f'| Long     | Quantity {quantity} | Ask Price {current_midprice:.2f} |')
                    print(f'| PnL {(prev_bid - current_midprice)*quantity:.2f} | Holdings {holdings} |')
                    print('\n')
                continue
                    
            cum_notionals.append(notional)
                
        if timestep % 100 == 0:
            None
            print(f'Notional: {notional}')
            print(f'PnL: {pnl:.2f}, Quantity: {quantity}')
            print(f'Holding: {holdings}')
            print(f'Fees: {total_fees:.2f}')

描述：

	余下来的就是具体的策略了 一般是5个模型左右同时跑然后看signal情况