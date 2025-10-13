import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# OPTIONAL: If you’re still using S4 from your previous code
# from models_s4.s4.s4d import S4D
# else comment out S4 if you don’t need it
##############################################################################

##############################################################################
# 1) S4 + Transformer combo (same as your original, if you want to keep)
##############################################################################
class PureAttentionModel(nn.Module):
    """
    A custom attention-only model (no LSTMs, S4, or convolution).
    It stacks multiple multi-head self-attention blocks in series,
    with feed-forward sub-layers, residual connections, and layer norms.
    """
    def __init__(
        self,
        input_size=38,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        output_size=1
    ):
        super().__init__()
        # 1) Project input [B, L, input_size] -> [B, L, d_model]
        self.input_proj = nn.Linear(input_size, d_model)

        # 2) Build N identical blocks of:
        #    [ MultiheadAttention + Residual + LayerNorm ] -> 
        #    [ FeedForward + Residual + LayerNorm ]
        self.attn_layers = nn.ModuleList([])
        self.attn_norms  = nn.ModuleList([])
        self.ff_layers   = nn.ModuleList([])
        self.ff_norms    = nn.ModuleList([])

        for _ in range(num_layers):
            # multi-head self-attention
            attn = nn.MultiheadAttention(embed_dim=d_model, 
                                         num_heads=nhead, 
                                         dropout=dropout, 
                                         batch_first=True)
            # feed-forward network
            ff = nn.Sequential(
                nn.Linear(d_model, d_model*4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model*4, d_model)
            )

            self.attn_layers.append(attn)
            self.attn_norms.append(nn.LayerNorm(d_model))
            self.ff_layers.append(ff)
            self.ff_norms.append(nn.LayerNorm(d_model))

        # 3) Final output linear layer
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x: [B, L, input_size]
        Returns: [B, output_size]
        """
        # (a) Project to d_model
        x = self.input_proj(x)  # => [B, L, d_model]

        # (b) Pass through each attention block
        for attn, attn_norm, ff, ff_norm in zip(
            self.attn_layers, self.attn_norms, self.ff_layers, self.ff_norms
        ):
            # --- Self-Attention Sub-Layer ---
            # Residual connection
            x_res = x
            # For self-attn in batch-first mode:
            #   query/key/value = x
            x_attn, _ = attn(x, x, x)   # [B, L, d_model]
            x = x_res + x_attn         # residual
            x = attn_norm(x)           # layer norm

            # --- Feed-Forward Sub-Layer ---
            x_res = x
            x_ff = ff(x)               # [B, L, d_model]
            x = x_res + x_ff           # residual
            x = ff_norm(x)             # layer norm

        # (c) Global average-pool over the time dimension
        x = x.mean(dim=1)  # [B, d_model]

        # (d) Final linear projection -> [B, output_size]
        x = self.fc_out(x)
        return x
class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.transformer_encoder(x)


class S4Model(nn.Module):
    """
    Minimal S4 model (reference only if you want to keep S4).
    """
    def __init__(
        self,
        d_input,
        d_output=1,
        d_model=36,
        n_layers=3,
        dropout=0.1,
        prenorm=True
    ):
        super().__init__()
        # If you don't have S4D, comment these lines or replace with your code
        from models_s4.s4.s4d import S4D
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x shape: [B, L, d_input]
        """
        x = self.encoder(x)                   # -> [B, L, d_model]
        x = x.transpose(-1, -2)               # -> [B, d_model, L]
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)   # -> [B, L, d_model]
        x = self.decoder(x)       # -> [B, L, d_model]
        return x


class S4Transformer(nn.Module):
    """
    Original S4+Transformer from your code (if you still want it).
    """
    def __init__(self, 
                 input_size=38, 
                 d_model=36, 
                 nhead=4, 
                 num_layers=3, 
                 output_size=1):
        super().__init__()
        self.s4 = S4Model(
            d_input=input_size,
            d_output=output_size,
            d_model=d_model,
            n_layers=num_layers,
            dropout=0.1,
            prenorm=True
        )
        self.transformer_encoder = TransformerEncoderLayerCustom(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x shape: [B, L, input_size]
        """
        x = self.s4(x)                        # -> [B, L, d_model]
        x = x.permute(1, 0, 2)                # -> [L, B, d_model]
        x = self.transformer_encoder(x)        # -> [L, B, d_model]
        x = x.permute(1, 0, 2)                # -> [B, L, d_model]
        x = x.mean(dim=1)                     # -> [B, d_model]
        x = self.fc_out(x)                    # -> [B, output_size]
        return x


##############################################################################
# 2) LSTM (a single LSTM, simpler than multi-stack)
##############################################################################
class LSTMModel(nn.Module):
    """
    A simple LSTM-based model for time-series regression.
    """
    def __init__(self, 
                 input_size=38,
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.1, 
                 output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [B, L, input_size]
        out, (h, c) = self.lstm(x)     # out: [B, L, hidden_size]
        out = out.mean(dim=1)         # mean-pool across time
        out = self.fc_out(out)        # -> [B, output_size]
        return out


##############################################################################
# 3) DeepLOB (fixed)
##############################################################################
##############################################################################
# 1) GRU (a simple GRU-based model for time-series regression)
##############################################################################
class GRUModel(nn.Module):
    """
    A simple GRU-based model for time-series regression.
    """
    def __init__(self, 
                 input_size=38,
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.1, 
                 output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: [B, L, input_size]
        """
        gru_out, h = self.gru(x)         # gru_out: [B, L, hidden_size]
                                         # h:       [num_layers, B, hidden_size]
        # Example: mean-pool across the time dimension
        out = gru_out.mean(dim=1)        # -> [B, hidden_size]
        out = self.fc_out(out)           # -> [B, output_size]
        return out

##############################################################################
# 2) GRU + (dot) Attention
##############################################################################
class GRUAttn(nn.Module):
    """
    GRU + (dot) attention on top of the final hidden state.
    """
    def __init__(self, 
                 input_size=38,
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.1, 
                 output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        # For the attention, we’ll transform the final hidden
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: [B, L, input_size]
        """
        gru_out, h = self.gru(x)               # gru_out: [B, L, hidden], h: [num_layers, B, hidden]
        query = h[-1]                          # final layer’s hidden => [B, hidden]
        query_t = self.attn_linear(query)      # linear transform => [B, hidden]

        # Compute attention scores: 
        # Dot product for each time step => [B, L, hidden] x [B, hidden, 1] => [B, L]
        attn_scores = torch.bmm(gru_out, query_t.unsqueeze(2)).squeeze(-1)  # [B, L]
        attn_weights = F.softmax(attn_scores, dim=1)                        # [B, L]

        # Weighted sum of gru_out
        context = gru_out * attn_weights.unsqueeze(-1)                      # [B, L, hidden]
        context = context.sum(dim=1)                                        # [B, hidden]
        context = self.dropout(context)

        out = self.fc_out(context)  # => [B, output_size]
        return out

##############################################################################
class DeepLOB(nn.Module):
    """
    DeepLOB architecture for limit order book data.
    
    NOTE: The original kernel (1,10) in conv_block3 requires input_size >= 10.
    If your feature dimension < 10, you must either:
      (a) pad up to 10, 
      (b) reduce the kernel to (1, F), 
      (c) or adapt the code below (see comment).
    """
    def __init__(self, 
                 input_size=38, 
                 seq_len=100,   # not strictly used unless you rely on shaping
                 num_lstm_hidden=64, 
                 output_size=1):
        super().__init__()
        # -- check
        if input_size < 10:
            raise ValueError(
                f"DeepLOB is using a (1x10) kernel. input_size={input_size} is too small."
                " Either pad your input features or reduce the kernel size."
            )

        # 1) CONV BLOCK 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
        )
        # 2) CONV BLOCK 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
        )
        # 3) CONV BLOCK 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 9)),  # 1x9
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
        )

        # Inception-like
        self.branch_conv1 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.branch_conv3 = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.branch_conv5 = nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(2, 0))
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.branch_pool_conv = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,  # 4 * 32
            hidden_size=num_lstm_hidden,
            batch_first=True,
            num_layers=1,
            bidirectional=False
        )
        self.fc_out = nn.Linear(num_lstm_hidden, output_size)

    def forward(self, x):
        """
        x shape: [B, L, input_size]
        """
        B, L, F = x.shape
        # (B, 1, L, F)
        x = x.unsqueeze(1)
        # Pass through conv blocks
        x = self.conv_block1(x)  # -> [B, 16, L, F//2], ...
        x = self.conv_block2(x)  # -> [B, 16, L, F//4], ...
        x = self.conv_block3(x)  # -> [B, 16, L, F//4-?]

        # Inception
        b1 = self.relu(self.branch_conv1(x))
        b3 = self.relu(self.branch_conv3(x))
        b5 = self.relu(self.branch_conv5(x))
        bp = self.branch_pool(x)
        bp = self.relu(self.branch_pool_conv(bp))

        # concat on channel dimension
        x = torch.cat([b1, b3, b5, bp], dim=1)  # [B, 32*4=128, L, ?]

        # The final dimension after "?" 
        # Typically the kernel is (3x1) so we'd expect ?==F' to be 1 or so
        # so let's just squeeze last dim if it's 1
        if x.shape[-1] == 1:
            x = x.squeeze(-1)  # -> [B, 128, L]
        else:
            # If it's bigger than 1, do a global avg pool across that dimension
            x = x.mean(dim=-1)  # -> [B, 128, L]

        # Transpose for LSTM
        x = x.transpose(1, 2)  # -> [B, L, 128]
        x, _ = self.lstm(x)    # -> [B, L, hidden]
        x = x.mean(dim=1)      # -> [B, hidden]
        x = self.fc_out(x)     # -> [B, output_size]
        return x


##############################################################################
# 4) LSTM + (dot) Attention
##############################################################################
class LSTMAttn(nn.Module):
    """
    LSTM + (dot) attention on top of final hidden state.
    """
    def __init__(self, 
                 input_size=38,
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.1, 
                 output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: [B, L, input_size]
        """
        lstm_out, (h, c) = self.lstm(x)   # [B, L, hidden], [2?, B, hidden]
        query = h[-1]                    # last layer’s hidden: [B, hidden]
        query_t = self.attn_linear(query)# [B, hidden]
        # Dot product for each time step: [B, L, hidden] x [B, hidden, 1] => [B, L, 1]
        attn_scores = torch.bmm(lstm_out, query_t.unsqueeze(2)).squeeze(-1)  # [B, L]
        attn_weights = F.softmax(attn_scores, dim=1)                         # [B, L]
        context = lstm_out * attn_weights.unsqueeze(-1)                      # [B, L, hidden]
        context = context.sum(dim=1)                                         # [B, hidden]
        context = self.dropout(context)
        out = self.fc_out(context)  # -> [B, output_size]
        return out


##############################################################################
# 5) Vanilla Transformer (purely transformer-based)
##############################################################################
class VanillaTransformer(nn.Module):
    """
    A simple, standard TransformerEncoder stack for time-series regression.
    """
    def __init__(self,
                 input_size=38,
                 d_model=64,
                 nhead=4,
                 num_layers=3,
                 dropout=0.1,
                 output_size=1):
        super().__init__()
        # A projection from input_size -> d_model
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x: [B, L, input_size]
        """
        # Project input
        x = self.input_proj(x)          # -> [B, L, d_model]
        # Transformer wants [L, B, d_model]
        x = x.permute(1, 0, 2)          # -> [L, B, d_model]
        x = self.transformer_encoder(x) # -> [L, B, d_model]
        # average pool time dimension
        x = x.permute(1, 0, 2)          # -> [B, L, d_model]
        x = x.mean(dim=1)               # -> [B, d_model]
        x = self.fc_out(x)              # -> [B, output_size]
        return x


##############################################################################
# 6) StackLSTM (Multiple LSTMs stacked in series)
##############################################################################
class StackLSTM(nn.Module):
    """
    A custom multi-stack LSTM approach:
      - LSTM1 -> LSTM2 -> ... -> LSTMn
      - Optionally feed each output to next input or just feed final hidden
    For simplicity, we chain them in series.
    """
    def __init__(self, 
                 input_size=38, 
                 hidden_sizes=[64, 64], 
                 dropout=0.1, 
                 output_size=1):
        super().__init__()
        # Build N LSTMs in a list
        self.lstm_layers = nn.ModuleList()
        in_size = input_size
        for hs in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hs,
                    num_layers=1,
                    dropout=0.0,       # Not using built-in dropout in each single-layer
                    batch_first=True
                )
            )
            in_size = hs
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(in_size, output_size)

    def forward(self, x):
        """
        x: [B, L, input_size]
        """
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, (h, c) = lstm(out)  # out: [B, L, hidden_size_i]
        # We can do a mean-pool or use last step
        out = out.mean(dim=1)     # [B, hidden_size_of_last_layer]
        out = self.dropout(out)
        out = self.fc_out(out)    # -> [B, output_size]
        return out

##############################################################################
# 6) LSTM_trans (Multiple LSTMs stacked in series)
##############################################################################

import torch
import torch.nn as nn

class TransEncLSTM(nn.Module):
    """
    A model that first applies a Transformer Encoder, then an LSTM.
    """
    def __init__(
        self,
        input_size=38,
        d_model=64,
        nhead=4,
        num_layers=3,          # Number of Transformer layers
        dropout=0.1,
        lstm_hidden=64,
        lstm_layers=1,         # Number of LSTM layers
        output_size=1
    ):
        super().__init__()
        # 1) Project input feature dimension -> d_model
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 2) Standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )
        
        # 3) LSTM after the transformer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        # 4) Final linear for regression/classification
        self.fc_out = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        """
        x: [B, L, input_size]
        """
        # (a) Project input -> [B, L, d_model]
        x = self.input_proj(x)  # => [B, L, d_model]
        
        # (b) Transformer wants [L, B, d_model]
        x = x.permute(1, 0, 2)  # => [L, B, d_model]
        x = self.transformer_encoder(x)  # => [L, B, d_model]

        # (c) Convert back to batch-first => [B, L, d_model]
        x = x.permute(1, 0, 2)

        # (d) Pass through LSTM => [B, L, lstm_hidden]
        x, (h, c) = self.lstm(x)

        # (e) Pool or use last hidden
        x = x.mean(dim=1)   # e.g. simple mean-pool

        # (f) Final linear => [B, output_size]
        x = self.fc_out(x)
        return x

def get_model(
    model_name: str,
    input_size=38,
    output_size=1,
    # for S4Transformer
    d_model=36,
    nhead=4,
    num_layers=3,
    # for LSTM and LSTMAttn/GRU
    hidden_size=64,
    dropout=0.1,
    # for DeepLOB
    num_lstm_hidden=64,
    # for StackLSTM
    hidden_sizes=[64, 64],
    attn_d_model=32,
    attn_nhead=4,
    attn_num_layers=1,
    # new TransEncLSTM options
    trans_lstm_hidden=64,
    trans_lstm_layers=1
):
    name = model_name.lower()

    # ------------------------------------------------------------------------
    # Existing models ...
    # ------------------------------------------------------------------------
    if name == "pure_attention":
        return PureAttentionModel(
            input_size=input_size,
            d_model=attn_d_model,
            nhead=attn_nhead,
            num_layers=attn_num_layers,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "s4transformer":
        return S4Transformer(
            input_size=input_size, 
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_size=output_size
        )
    elif name == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "deeplob":
        return DeepLOB(
            input_size=input_size,
            seq_len=100,  # or adapt as needed
            num_lstm_hidden=num_lstm_hidden,
            output_size=output_size
        )
    elif name == "lstm_attn":
        return LSTMAttn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "vanilla_transformer":
        return VanillaTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "stacklstm":
        return StackLSTM(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "trans_enc_lstm":
        return TransEncLSTM(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            lstm_hidden=trans_lstm_hidden,
            lstm_layers=trans_lstm_layers,
            output_size=output_size
        )

    # ------------------------------------------------------------------------
    # NEW GRU-based models
    # ------------------------------------------------------------------------
    elif name == "gru":
        return GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    elif name == "gru_attn":
        return GRUAttn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Choose from: s4transformer, lstm, deeplob, lstm_attn, vanilla_transformer, "
            f"stacklstm, trans_enc_lstm, gru, gru_attn"
        )