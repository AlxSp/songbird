from songbird.nn.blocks.Conv2d import DResidualBlockConv2d, ResizeResidualBlockConv2d

import math
from turtle import forward
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stem
        self.stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        # Down Part
        self.res_block1 = DResidualBlockConv2d(in_channels=32, out_channels=32, stride=2)
        self.res_block2 = DResidualBlockConv2d(in_channels=32, out_channels=32)
        
        self.res_block3 = DResidualBlockConv2d(in_channels=32, out_channels=64, stride=2)
        self.res_block4 = DResidualBlockConv2d(in_channels=64, out_channels=64)
        
        self.res_block5 = DResidualBlockConv2d(in_channels=64, out_channels=128, stride=2)
        self.res_block6 = DResidualBlockConv2d(in_channels=128, out_channels=128)
        
        # Middle Part
        self.res_block7 = DResidualBlockConv2d(in_channels=128, out_channels=256)
        self.res_block8 = DResidualBlockConv2d(in_channels=256, out_channels=256)
        
        # Up Part 
        self.res_block9 = ResizeResidualBlockConv2d(in_channels=256, out_channels=128, scale_factor=2)
        self.res_block10 = DResidualBlockConv2d(in_channels=256, out_channels=128)
        
        self.res_block11 = ResizeResidualBlockConv2d(in_channels=128, out_channels=64, scale_factor=2)
        self.res_block12 = DResidualBlockConv2d(in_channels=128, out_channels=64)
        
        self.res_block13 = ResizeResidualBlockConv2d(in_channels=64, out_channels=32, scale_factor=2)
        self.res_block14 = DResidualBlockConv2d(in_channels=64, out_channels=32)
        # Head
        self.head = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.stem(x)
        
        x = self.res_block1(x)
        conv1 = self.res_block2(x)
        
        x = self.res_block3(conv1)
        conv2 = self.res_block4(x)
        
        x = self.res_block5(conv2)
        conv3 = self.res_block6(x)
        
        
        x = self.res_block7(conv3)
        x = self.res_block8(x)
        
        
        x = self.res_block9(x)
        x = self.res_block10(torch.cat((x, conv3), dim=1))
        
        x = self.res_block11(x)
        x = self.res_block12(torch.cat((x, conv2), dim=1))
        
        x = self.res_block13(x)
        x = self.res_block14(torch.cat((x, conv1), dim=1))
        
        x = self.head(x)
        
        return x