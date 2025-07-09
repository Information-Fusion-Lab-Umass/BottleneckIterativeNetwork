import models.hyperformer as hf
import torch.nn as nn
import torch


class CrossAttention1D(nn.Module):
    def __init__(self, d_model, d_aux, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, kdim=d_aux, vdim=d_aux, batch_first=True)
        self.norm1 = hf.LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = hf.PositionwiseFeedForward(d_model=d_model, hidden=dim_feedforward, dropout=dropout)
        self.norm2 = hf.LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, aux, src_mask=None):
        # 1. compute self attention
        _x = x

        x, _ = self.attention(query=x, key=aux, value=aux, attn_mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


