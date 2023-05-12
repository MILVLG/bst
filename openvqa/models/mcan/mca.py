
from openvqa.ops.slimmable_ops import SlimmableLinear, SlimmableLayerNorm, SlimmableMLP

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = SlimmableLinear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = SlimmableLinear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = SlimmableLinear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = SlimmableLinear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, width):
        n_batches = q.size(0)

        v = self.linear_v(v, width).view(
            n_batches,
            -1,
            int(self.__C.MULTI_HEAD * width),
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k, width).view(
            n_batches,
            -1,
            int(self.__C.MULTI_HEAD * width),
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q, width).view(
            n_batches,
            -1,
            int(self.__C.MULTI_HEAD * width),
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            int(self.__C.HIDDEN_SIZE * width)
        )

        atted = self.linear_merge(atted, width)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = SlimmableMLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x, width):
        return self.mlp(x, width)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = SlimmableLayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = SlimmableLayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask, width):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask, width)
        ), width)

        y = self.norm2(y + self.dropout2(
            self.ffn(y, width)
        ), width)

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = SlimmableLayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = SlimmableLayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = SlimmableLayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, width):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask, width=width)
        ), width)

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask, width=width)
        ), width)

        x = self.norm3(x + self.dropout3(
            self.ffn(x, width)
        ), width)

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.__C = __C
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.enc_priority_idx = [0, 5, 1, 4, 2, 3]
        self.dec_priority_idx = [5, 0, 1, 4, 2, 3]

    def forward(self, y, x, y_mask, x_mask, width, depth):
        cur_enc_idx = sorted(self.enc_priority_idx[:int(depth * self.__C.LAYER)])
        cur_dec_idx = sorted(self.dec_priority_idx[:int(depth * self.__C.LAYER)])

        for idx in cur_enc_idx:
            y = self.enc_list[idx](y, y_mask, width)

        for idx in cur_dec_idx:
            x = self.dec_list[idx](x, y, x_mask, y_mask, width)

        return y, x
