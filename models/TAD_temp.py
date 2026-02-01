import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class Tadtempo(nn.Module):
    def __init__(self, num_layers, num_filter_maps, dim):
        super(Tadtempo, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_filter_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_filter_maps, num_filter_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_filter_maps, num_filter_maps, 1)

    def forward(self, x, mask):
        
        out = self.conv_1x1(x)
        
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


