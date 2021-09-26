import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def train_on_mnist():

    class MnistEncoder(nn.Module):
        def __init__(self):
            super(MnistEncoder, self).__init__()
            
            #self.conv1 = nn.Conv1d(1, 6, 3)
            self.dense1 = nn.Linear(784, 400) 
        # self.dense2 = nn.Linear(256, 128) 
            self.mean_dense = nn.Linear(400, 20)
            self.variance_dense = nn.Linear(400, 20)

        def forward(self, x):
            x =  F.relu(self.dense1(x))
            #x =  F.relu(self.dense2(x))
            x_mean =  self.mean_dense(x)
            x_variance =  self.variance_dense(x)
            return  x_mean, x_variance

    class MnistDecoder(nn.Module):
        def __init__(self):
            super(MnistDecoder, self).__init__()
            
            #self.conv1 = nn.Conv1d(1, 6, 3)
            self.dense1 = nn.Linear(20, 400)
            #self.dense2 = nn.Linear(128, 256) 
            self.dense3 = nn.Linear(400, 784) 

        def forward(self, x):
            x =  F.relu(self.dense1(x))
            #x =  F.relu(self.dense2(x))
            x =  F.sigmoid(self.dense3(x))
            return x  




    model_name = "mnist_model"
    report_dir = os.path.join(ap.project_base_dir, "reports", "vae", model_name)
    ap.empty_or_create_dir(report_dir)
    
    set_torch_seeds(42)

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
                    transforms.Lambda(lambda x: torch.flatten(x))
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
                    transforms.Lambda(lambda x: torch.flatten(x))
                ]
            )
        ),
        batch_size=batch_size, shuffle=True, **kwargs)


    encoder = MnistEncoder()
    decoder = MnistDecoder()
    model = vae.VariationalAutoDecoder(encoder, decoder)

    plot_params = {
        "report_dir" : report_dir,
        "plot_num" : 4
    }


    learning_rate = 1e-4
    epochs = 100

    train_plus(model, train_loader, test_loader, loss_function, learning_rate, epochs, show_mnist, plot_params)

    print()
    print()