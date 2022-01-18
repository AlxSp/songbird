import torch
import torch.nn as nn
import torch.nn.functional as F

from songbird.nn.blocks.Conv2d import DResidualBlockConv2d, ResizeResidualBlockConv2d, ResidualBlockConv2d

#%%
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        self.stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        
        self.stage_1 = nn.Sequential(
            DResidualBlockConv2d(in_channels=32, out_channels=32, stride=2),
            DResidualBlockConv2d(in_channels=32, out_channels=32)
        )
        
        self.stage_2 = nn.Sequential(
            DResidualBlockConv2d(in_channels=32, out_channels=64, stride=2),
            DResidualBlockConv2d(in_channels=64, out_channels=64)
        )
        
        self.stage_3 = nn.Sequential(
            DResidualBlockConv2d(in_channels=64, out_channels=128, stride=2),
            DResidualBlockConv2d(in_channels=128, out_channels=128)
        )
        
        self.stage_4 = nn.Sequential(
            DResidualBlockConv2d(in_channels=128, out_channels=256, stride=2),
            DResidualBlockConv2d(in_channels=256, out_channels=256)
        )
        
        self.mean_dense = nn.Linear(4096, 256)
        self.variance_dense = nn.Linear(4096, 256)

    def forward(self, x):
        
        x = self.stem(x)
        
        x = self.stage_1(x)
        
        x = self.stage_2(x)
        
        x = self.stage_3(x)
        
        x = self.stage_4(x)
        # print(f"Input shape: {x.shape}")
        # x = self.res_block1(x)#F.relu(self.conv1(x))
        # x = self.res_block2(x)#F.relu(self.conv2(x))
        
        # x = self.res_block3(x)#F.relu(self.conv3(x))
        # x = self.res_block4(x)#F.relu(self.conv3(x))
        # # print(f"Res 3 output shape: {x.shape}")
        # # x = self.max_pool3(x)
        # x = self.res_block5(x)#F.relu(self.conv3(x))
        # x = self.res_block6(x)#F.relu(self.conv3(x))
        
        # x = self.res_block7(x)
        # x = self.res_block8(x)

        batch_size = x.shape[0]
        # print(f"x shape: {x.shape}")
        x = x.reshape(batch_size, -1)

        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        
        return  x_mean, x_variance

#%%
class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        self.dense1 = nn.Linear(256, 4096)
        
        # self.stem = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.stage_1 = nn.Sequential(
            ResizeResidualBlockConv2d(in_channels=256, out_channels=256, scale_factor=2),
            ResidualBlockConv2d(in_channels=256, out_channels=128)
        )
        
        self.stage_2 = nn.Sequential(
            ResizeResidualBlockConv2d(in_channels=128, out_channels=128, scale_factor=2),
            ResidualBlockConv2d(in_channels=128, out_channels=64)
        )
        
        self.stage_3 = nn.Sequential(
            ResizeResidualBlockConv2d(in_channels=64, out_channels=64, scale_factor=2),
            ResidualBlockConv2d(in_channels=64, out_channels=32)
        )
        
        self.stage_4 = nn.Sequential(
            ResizeResidualBlockConv2d(in_channels=32, out_channels=32, scale_factor=2),
            ResizeResidualBlockConv2d(in_channels=32, out_channels=32)
        )
        
        self.head = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.dense1(x)
        x = x.view(-1, 256, 16, 1)

        x = self.stage_1(x)
        
        x = self.stage_2(x)
        
        x = self.stage_3(x)
        
        x = self.stage_4(x)
        
        # x = self.res_block1(x)
        # x = self.res_block2(x)
        
        # x = self.res_block3(x)
        # x = self.res_block4(x)
        
        # x = self.res_block5(x)
        # x = self.res_block6(x)
        
        # x = self.res_block7(x)
        # x = self.res_block8(x)
        
        x = self.head(x)
        
        x = torch.sigmoid(x)

        return x  
#%%
class VariationalAutoEncoder(nn.Module):  
    def __init__(self, encoder, decoder):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x_mean, x_variance = self.encoder(x)

        if self.training:
            std = torch.exp(x_variance * 0.5)
            eps = torch.randn_like(std) 
            z = x_mean + (eps * std)
        else:
            z = x_mean

        x = self.decoder(z)
        return x, x_mean, x_variance
