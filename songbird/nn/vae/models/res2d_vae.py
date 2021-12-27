import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockConv2d, self).__init__()
        # bias is set to False because we are using batch normalization
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
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
#%%
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        self.stem = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)

        self.res_block1 = ResidualBlockConv2d(in_channels=32, out_channels=32, stride=2)
        self.res_block2 = ResidualBlockConv2d(in_channels=32, out_channels=32)
        
        
        self.res_block3 = ResidualBlockConv2d(in_channels=32, out_channels=64, stride=2)
        self.res_block4 = ResidualBlockConv2d(in_channels=64, out_channels=64)
        
        self.res_block5 = ResidualBlockConv2d(in_channels=64, out_channels=128, stride=2)
        self.res_block6 = ResidualBlockConv2d(in_channels=128, out_channels=128)

        # self.res_block7 = ResidualBlockConv2d(in_channels=64, out_channels=256, stride=2)
        # self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.mean_dense = nn.Linear(8192, 256)
        self.variance_dense = nn.Linear(8192, 256)

    def forward(self, x):
        
        x = self.stem(x)
        # print(f"Input shape: {x.shape}")
        x = self.res_block1(x)#F.relu(self.conv1(x))
        # print(f"Res 1 output shape: {x.shape}")
        # x = self.max_pool1(x)
        x = self.res_block2(x)#F.relu(self.conv2(x))
        # print(f"Res 2 output shape: {x.shape}")
        # x = self.max_pool2(x)
        
        x = self.res_block3(x)#F.relu(self.conv3(x))
        x = self.res_block4(x)#F.relu(self.conv3(x))
        # print(f"Res 3 output shape: {x.shape}")
        # x = self.max_pool3(x)
        x = self.res_block5(x)#F.relu(self.conv3(x))
        x = self.res_block6(x)#F.relu(self.conv3(x))

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        
        return  x_mean, x_variance

class ResizeResidualBlockConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale_factor=2):
        super(ResizeResidualBlockConv2d, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
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
#%%
class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        self.dense1 = nn.Linear(256, 8192)
        
        # self.stem = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)

        self.res_block1 = ResizeResidualBlockConv2d(in_channels=128, out_channels=128, scale_factor=2)
        self.res_block2 = ResidualBlockConv2d(in_channels=128, out_channels=64)
        
        
        self.res_block3 = ResizeResidualBlockConv2d(in_channels=64, out_channels=64, scale_factor=2)
        self.res_block4 = ResidualBlockConv2d(in_channels=64, out_channels=32)
        
        self.res_block5 = ResizeResidualBlockConv2d(in_channels=32, out_channels=32, scale_factor=2)
        self.res_block6 = ResizeResidualBlockConv2d(in_channels=32, out_channels=32)
        
        self.head = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv5 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        x = self.dense1(x)
        x = x.view(-1, 128, 32, 2)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        x = self.head(x)
        
        x = torch.sigmoid(x)
        # print(f"Input shape: {x.shape}")
        # x =  F.relu(self.dense1(x))
        # # print(f"Dense output shape: {x.shape}")
        # x = x.view(-1, 128, 32, 2)
        # # print(f"Reshape shape: {x.shape}")
        # x = F.relu(self.conv1(x))
        # # print(f"Conv 1 output shape: {x.shape}")
        # x = F.relu(self.conv2(x))
        # # print(f"Conv 2 output shape: {x.shape}")
        # x = F.relu(self.conv3(x))
        # # print(f"Conv 4 output shape: {x.shape}")
        # x = F.relu(self.conv4(x))
        # # print(f"Conv 4 output shape: {x.shape}")
        # x =  torch.sigmoid(self.conv5(x))
        # print(f"Conv 5 output shape: {x.shape}")
        #x = x.view(-1, 64, 1, 1)
        #x =  F.relu(self.dense2(x))
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
