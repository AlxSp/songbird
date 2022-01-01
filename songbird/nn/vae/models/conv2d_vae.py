import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
                
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # self.conv11 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        self.mean_dense = nn.Linear(4096, 256)
        self.variance_dense = nn.Linear(4096, 256)

    def forward(self, x):
        
        
        x = self.stem(x)
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"Conv 1 output shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"Conv 2 output shape: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"Conv 3 output shape: {x.shape}")
        x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        x = F.relu(self.conv5(x))
        
        x = F.relu(self.conv6(x))
        
        x = F.relu(self.conv7(x))
        
        x = F.relu(self.conv8(x))
        
        # x = F.relu(self.conv9(x))
        
        # x = F.relu(self.conv10(x))
        
        # x = F.relu(self.conv11(x))
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
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.dense1 = nn.Linear(256, 4096)
        
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.head = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
                
        # self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.conv5 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        x =  F.relu(self.dense1(x))
        # print(f"Dense output shape: {x.shape}")
        x = x.view(-1, 256, 16, 1)
                
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # print(f"Conv 4 output shape: {x.shape}")
        
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.upsample(x)
        x = self.conv7(x) 
        x = self.conv8(x)
        
        x = self.upsample(x)
        x = self.head(x)
        # print(f"Conv 5 output shape: {x.shape}")
        #x = x.view(-1, 64, 1, 1)
        #x =  F.relu(self.dense2(x))
        x = F.sigmoid(x)
        
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