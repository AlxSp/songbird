import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBottleneckBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ResidualBottleneckBlockConv2d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm3 = nn.BatchNorm2d(num_features=out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
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
    
class DResidualBottleneckBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ResidualBottleneckBlockConv2d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm3 = nn.BatchNorm2d(num_features=out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
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


class ResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ResidualBlockConv2d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        #if the stride is not 1 or the input and output channels are not the same, transform residual value into the right shape
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
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
    
class DResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(DResidualBlockConv2d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        #if the stride is not 1 or the input and output channels are not the same, transform residual value into the right shape
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
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
    
class ResizeResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, scale_factor=2):
        super(ResizeResidualBlockConv2d, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        #if the stride is not 1 or the input and output channels are not the same, transform residual value into the right shape
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        x = self.upsample(x)
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)
        
        return x

class TransposeResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,):
        super(TransposeResidualBlockConv2d, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residual = nn.Identity()


    def forward(self, x):
        residual =  x
        x = self.tconv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.tconv2(x)
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)

        return x