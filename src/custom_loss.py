import numpy as np 
import torch

def neg_log_likelihood(Y, mu, sigma):
    N = Y.shape[0]
    log_likelihood = - (1/2) * torch.log(2*np.pi*sigma**2)  - (1/N)* torch.sum((Y-mu)**2/(2*sigma**2) , axis=0)
    return -torch.mean(log_likelihood)


def neg_log_likelihood_loss(Y_pred, Y):
    mu = Y_pred[:, 0, :]
    std = Y_pred[:, 1, :]
    return neg_log_likelihood(Y, mu, std)



