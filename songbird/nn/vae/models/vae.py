import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.dense1 = nn.Linear(16384, 10384)
        self.dense2 = nn.Linear(10384, 8384)
        self.dense3 = nn.Linear(8384, 6384)

        self.mean_dense = nn.Linear(6384, 256)
        self.variance_dense = nn.Linear(6384, 256)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.dense1(x))
        # print(f"Conv 1 output shape: {x.shape}")
        x = F.relu(self.dense2(x))
        # print(f"Conv 2 output shape: {x.shape}")
        x = F.relu(self.dense3(x))
        # print(f"Conv 3 output shape: {x.shape}")
        # print(f"Conv 4 output shape: {x.shape}")
        # print(f"Conv 5 output shape: {x.shape}")

        # x = self.pool(x)
        # print(f"Pool output shape: {x.shape}")

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
        
        self.mean_dense = nn.Linear(256, 6384)
        self.dense1 = nn.Linear(6384, 8384)
        self.dense2 = nn.Linear(8384, 10384)
        self.dense3 = nn.Linear(10384, 16384)

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        x =  F.relu(self.mean_dense(x))
        x =  F.relu(self.dense1(x))
        x =  F.relu(self.dense2(x))
        x =  F.sigmoid(self.dense3(x))
        x = x.view(x.size(0),1, 512, 32)
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