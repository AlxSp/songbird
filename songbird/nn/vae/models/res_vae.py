import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2d, self).__init__()
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

class TransposeResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,):
        super(TransposeResidualBlock2d, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

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
class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()

        self.res_block1 = ResidualBlock2d(in_channels=1, out_channels=16)
        self.res_block2 = ResidualBlock2d(in_channels=16, out_channels=32)
        self.res_block3 = ResidualBlock2d(in_channels=32, out_channels=64)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        #self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        #self.dense1 = nn.Linear(4410, 400) 
        
        self.mean_dense = nn.Linear(16384, 256)
        self.variance_dense = nn.Linear(16384, 256)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.res_block1(x)#F.relu(self.conv1(x))
        # print(f"Res 1 output shape: {x.shape}")
        x = self.max_pool1(x)

        x = self.res_block2(x)#F.relu(self.conv2(x))
        # print(f"Res 2 output shape: {x.shape}")
        x = self.max_pool2(x)
        
        x = self.res_block3(x)#F.relu(self.conv3(x))
        # print(f"Res 3 output shape: {x.shape}")
        x = self.max_pool3(x)
        #x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        #x = F.relu(self.conv5(x))
        # print(f"Conv 5 output shape: {x.shape}")

        # x = self.pool(x)
        # print(f"Pool output shape: {x.shape}")
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        # print(f"Flattened output shape: {x.shape}")

        #x = F.adaptive_avg_pool2d(x, 3).reshape(batch_size, -1)
        #print(f"Flattened output shape: {x.shape}")
        # x = self.dense(x)

        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        
        return  x_mean, x_variance
#%%
class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        self.dense1 = nn.Linear(256, 4096)
        #self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        x =  F.relu(self.dense1(x))
        # print(f"Dense output shape: {x.shape}")
        x = x.view(-1, 64, 2, 32) #x.view(-1, 128, 1, 16)
        # print(f"Reshape shape: {x.shape}")
        #x = F.relu(self.conv1(x))
        # print(f"Conv 1 output shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"Conv 2 output shape: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x =  F.sigmoid(self.conv5(x))
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
