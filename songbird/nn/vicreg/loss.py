import torch 
import torch.nn.functional as F

def invariance_loss(x, y):
    return F.mse_loss(x, y)

def variance_loss(x, y, gamma = 1.0, epsilon = 0.0001):
    """
    Variance loss function.
    """

    # compute the standard deviation of x and y along the batch dimension. Epsilon is added to avoid numerical instability
    std_x = torch.sqrt(torch.var(x, dim=0) + epsilon) 
    std_y = torch.sqrt(torch.var(y, dim=0) + epsilon)
    
    return torch.mean(F.relu(gamma - std_x)) + torch.mean(F.relu(gamma - std_y))

def covariance_loss(x, y):
    batch_size, num_embeddings = x.shape

    z_x = x - torch.mean(x, dim=0)
    z_y = y - torch.mean(y, dim=0)

    cov_z_x = (torch.matmul(z_x.T, z_x) / (batch_size - 1)).square()
    cov_z_y = (torch.matmul(z_y.T, z_y) / (batch_size - 1)).square()

    loss_c_x = (cov_z_x.sum() - cov_z_x.diagonal().sum()) / num_embeddings
    loss_c_y = (cov_z_y.sum() - cov_z_y.diagonal().sum()) / num_embeddings

    return loss_c_x + loss_c_y

def combine_losses(invariance_loss, variance_loss, covariance_loss, invariance_weight = 25, variance_weight = 25, covariance_weight = 1):
    return invariance_weight * invariance_loss + variance_weight * variance_loss + covariance_weight * covariance_loss


