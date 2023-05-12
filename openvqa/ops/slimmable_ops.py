
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_num, out_features_num, style="mid", bias=True):
        super(SlimmableLinear, self).__init__(
            in_features_num, out_features_num, bias=bias)

        self.in_features_num = in_features_num
        self.out_features_num = out_features_num
        self.style = style

    def forward(self, input, width):
        if self.style == "start":
            return nn.functional.linear(input, self.weight[:int(self.out_features_num * width), :], self.bias[:int(self.out_features_num * width)])
        elif self.style == "end":
            return nn.functional.linear(input, self.weight[:, :int(self.in_features_num * width)], self.bias)
        elif self.style == "mid":
            return nn.functional.linear(input, self.weight[:int(self.out_features_num * width), :int(self.in_features_num * width)], self.bias[:int(self.out_features_num * width)])


class SlimmableLayerNorm(nn.Module):
    def __init__(self, maxsize, eps=1e-6):
        super(SlimmableLayerNorm, self).__init__()

        self.maxsize = maxsize
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(maxsize))
        self.bias = nn.Parameter(torch.zeros(maxsize))

    def forward(self, x, width):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight[: int(width * self.maxsize)] * (x - mean) / (std + self.eps) + self.bias[: int(width * self.maxsize)]


class SlimmableFC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(SlimmableFC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = SlimmableLinear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x, width):
        x = self.linear(x, width)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class SlimmableMLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(SlimmableMLP, self).__init__()

        self.fc = SlimmableFC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = SlimmableLinear(mid_size, out_size)

    def forward(self, x, width):
        return self.linear(self.fc(x, width), width)


class SlimmableAttFlat(nn.Module):
    def __init__(self, __C):
        super(SlimmableAttFlat, self).__init__()
        self.__C = __C

        self.fc = SlimmableFC(__C.HIDDEN_SIZE, __C.FLAT_MLP_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)
        self.linear = SlimmableLinear(__C.FLAT_MLP_SIZE, __C.FLAT_GLIMPSES, style='end')
        self.linear_merge = SlimmableLinear(int(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES), __C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask, width):
        att = self.linear(self.fc(x, width), width)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted, width)

        return x_atted
