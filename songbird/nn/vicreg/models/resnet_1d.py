import torch
import torch.nn as nn
import torch.nn.functional as F

from songbird.nn.blocks.Conv1d import DResidualBlockConv1d

class Resnet1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_growth_rate, dilation_depth, repeat_num, kernel_size):
        super(Resnet1D, self).__init__()
        
        dilations = [dilation_growth_rate ** d for d in range(dilation_depth)] * repeat_num

        self.model = nn.Sequential([
            DResidualBlockConv1d(in_channels=in_channels, 
                                 out_channels=out_channels, 
                                 stride=1, 
                                 groups=1, 
                                 kernel_size=kernel_size, 
                                 dilation=dilation)
            for dilation in dilations])

    def forward(self, x):
        return self.model(x)