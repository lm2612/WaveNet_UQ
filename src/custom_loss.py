import torch


def neg_log_likelihood(Y, mu, sigma):
    return -torch.sum((Y-mu)**2/(2*sigma**2))

def neg_log_likelihood_loss(Y_pred, Y):
    mu = Y_pred[:, 0, :]
    std = Y_pred[:, 1, :]
    return neg_log_likelihood(Y, mu, std)
