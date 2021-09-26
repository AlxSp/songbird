import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(4410, 400) 
       # self.dense2 = nn.Linear(256, 128) 
        self.mean_dense = nn.Linear(400, 20)
        self.variance_dense = nn.Linear(400, 20)

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        #x =  F.relu(self.dense2(x))
        x_mean =  self.mean_dense(x)
        x_variance =  self.variance_dense(x)
        return  x_mean, x_variance

class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(20, 400)
        #self.dense2 = nn.Linear(128, 256) 
        self.dense3 = nn.Linear(400, 4410) 

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        #x =  F.relu(self.dense2(x))
        x =  F.sigmoid(self.dense3(x))
        return x  

class VariationalAutoDecoder(nn.Module):  
    def __init__(self, encoder, decoder):
        super(VariationalAutoDecoder, self).__init__()

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

def loss_function(x_hat, x, mean, variance):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + KLD


def mse_loss_function(x_hat, x, mean, variance):
    BCE = nn.functional.mse_loss(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + KLD