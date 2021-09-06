# %%
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import songbird.audio_events.audio_processing as ap

import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# %%
class MnistEncoder(nn.Module):
    def __init__(self):
        super(MnistEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        #self.conv1 = nn.Conv1d(1, 6, 3)
        #self.dense = nn.Linear(12800, 1568) 
        #self.dense2 = nn.Linear(256, 128) 
        self.mean_dense = nn.Linear(12800, 128)
        self.variance_dense = nn.Linear(12800, 128)

    def forward(self, x):
        x =  F.relu(self.conv1(x))
        # print(f"Conv 1) x shape: {x.shape}")

        x =  F.relu(self.conv2(x))
        # print(f"Conv 2) x shape: {x.shape}")

        x = x.view(x.size(0), -1)
        # print(f"Reshape 1) x shape: {x.shape}")

        # x =  F.relu(self.dense(x))
        # print(f"Dense 1) x shape: {x.shape}")

        x_mean =  self.mean_dense(x)
        # print(f"Mean and Var) x shape: {x_mean.shape}")

        x_variance =  self.variance_dense(x)
        return  x_mean, x_variance

# %%
class MnistDecoder(nn.Module):
    def __init__(self):
        super(MnistDecoder, self).__init__()
        
        #self.conv1 = nn.Conv1d(1, 6, 3)
        self.dense1 = nn.Linear(128, 12800)
        # self.dense2 = nn.Linear(1568, 3136) 

        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.conv2 = nn.ConvTranspose2d(16, 1, kernel_size=5)
        #self.dense2 = nn.Linear(128, 256) 

    def forward(self, x):
        # print(f"Input) x shape: {x.shape}")
        x =  F.relu(self.dense1(x))
        # print(f"Dense 1) x shape: {x.shape}")
        # x =  F.relu(self.dense2(x))
        # print(f"Dense 2) x shape: {x.shape}")

        x = x.view(-1, 32, 20, 20)
        # print(f"Reshape 1) x shape: {x.shape}")

        x =  F.relu(self.conv1(x))
        # print(f"Conv 1) x shape: {x.shape}")
        
        x =  F.sigmoid(self.conv2(x))
        # print(f"Conv 2) x shape: {x.shape}")

        return x  

# %%
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
# %%
def loss_function(x_hat, x, mean, variance):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + KLD
# %%
def set_torch_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
# %%
def show_mnist(x, y, label=None, fig_num = 1, plot_num = 4, report_dir = None):
    for fig_index in range(fig_num):
        fig, axs = plt.subplots(2, 4)
        in_pic = x.data.cpu().view(-1, 28, 28)
        #axs.suptitle(label + ' - real test data / reconstructions', color='w', fontsize=16)
        for i in range(plot_num):
            axs[0, i].imshow(in_pic[fig_index * plot_num + i])#plt.subplot(1, plot_num, i + 1)
            axs[0, i].axis('off')
            #plt.imshow(in_pic[i+plot_num+plot_index])
            #plt.axis('off')
        out_pic = y.data.cpu().view(-1, 28, 28)
        #plt.figure(figsize=(18,6))
        for i in range(plot_num):
            #plt.subplot(1, plot_num, i + 1)
            axs[1, i].imshow(out_pic[fig_index * plot_num + i])
            axs[1, i].axis('off')

        fig.canvas.set_window_title(f"{label} - real test data / reconstructions'")
        plt.savefig(os.path.join(report_dir, f'{label}_{fig_index}.png'))



# %%
model_name = "mnist_model"
report_dir = os.path.join(ap.project_base_dir, "reports", "vae", model_name)
ap.empty_or_create_dir(report_dir)
# %%
set_torch_seeds(42)
# %%
batch_size = 256
kwargs = { 'num_workers': 1, 'pin_memory': True }

train_loader = torch.utils.data.DataLoader(
    MNIST(
        './data',
        train=True, 
        download=True, 
        transform= transforms.Compose(
            [
                transforms.ToTensor(), 
                #transforms.Lambda(lambda x: torch.flatten(x))
            ]
        )
    ),
    batch_size=batch_size,
    shuffle=True, 
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transforms.Compose(
            [
                transforms.ToTensor(), 
                #transforms.Lambda(lambda x: torch.flatten(x))
            ]
        )
    ),
    batch_size=batch_size, shuffle=True, **kwargs)

# %%
encoder = MnistEncoder()
decoder = MnistDecoder()
model = VariationalAutoDecoder(encoder, decoder)

plot_params = {
    "report_dir" : report_dir,
    "plot_num" : 4
}

# %%
learning_rate = 1e-4
epochs = 100
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device for training")
model.to(device)
model.train()
# %%
#codes = dict(mu = list(), variance = list(), y=list())
#input processing variables
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#general variables
for epoch in range(0, epochs + 1):
    if epoch > 0:
        model.train()
        train_loss = 0
        for x, _ in train_loader:

            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch:4} | train loss: {train_loss / len(train_loader.dataset):10.6f}", end = "\r")

    plot_x = None
    plot_x_hat = None
    means, variance, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x, y in test_loader:
                
            x = x.to(device)
            x_hat, mu, logvar = model(x)

            test_loss += loss_function(x_hat, x, mu, logvar)

            means.append(mu.detach())
            variance.append(logvar.detach())
            labels.append(y.detach())

            if len(x) > plot_params["plot_num"]:
                plot_x = x
                plot_x_hat = x_hat
    
    show_mnist(plot_x, plot_x_hat, f'Epoch_{epoch}', 1, **plot_params)



    