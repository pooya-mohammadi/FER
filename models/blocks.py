from torch import nn
import torch
from torch.nn import functional as F


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), padding=(0, 0),
                            bias=False)

    def forward(self, inputs):
        return self.op(inputs)


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=(1, 1), padding=(0, 0), bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        g = g.unsqueeze(-1).unsqueeze(-1)
        c = self.op(l + g)  # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, W, H), g
