import torch
import torch.nn as nn
import torch.nn.functional as F

class DResidualBottleneckBlockConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(DResidualBottleneckBlockConv1d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False) 
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm1d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm1d(num_features=out_channels)
        self.batch_norm3 = nn.BatchNorm1d(num_features=out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(num_features=out_channels)
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = x + residual
        x = F.relu(x)
        
        return x

class DResidualBlockConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, groups=1, dilation=1):
        super(DResidualBlockConv1d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=groups, dilation=dilation, bias=False) 
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        #if the stride is not 1 or the input and output channels are not the same, transform residual value into the right shape
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2) if stride != 1 else nn.Identity(),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residual = nn.Identity()
        

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        # print(f"Conv 1 output shape: {x.shape}")
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        # print(f"Conv 2 output shape: {x.shape}")
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)

        return x