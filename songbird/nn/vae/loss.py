import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(x_hat, x, mean, variance, beta = 1.0):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + beta * KLD


def mse_loss_function(x_hat, x, mean, variance, beta = 1.0):
    BCE = nn.functional.mse_loss(
        x_hat, x, reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(torch.exp(variance) - variance - 1 + mean.pow(2))

    return BCE + beta * KLD