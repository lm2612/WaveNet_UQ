import numpy as np
import torch
import torch.nn as nn

def check_nc(filename):
    """Check filename returns .nc file, otherwise add .nc"""
    if filename[-3:]!=".nc":
        return f"{filename}.nc"
    else:
        return filename

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)

def init_kaiming(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_uniform_(m.weight)

def count_parameters(model):
    """Returns total number of model parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
        else:
            m.eval()

def get_best_epoch(model_dir, validation_losses_filename="wavenet_validation_losses.csv", min_epochs=10):
    validation_losses = np.loadtxt(f"{model_dir}{validation_losses_filename}")
    best_epoch = np.argmin(validation_losses)
    if best_epoch <= min_epochs:
        return best_epoch
    else:
        return (min_epochs + np.argmin(validation_losses[min_epochs:]) ) 


