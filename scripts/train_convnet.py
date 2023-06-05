import os 
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random

from torch.nn import MSELoss, Module
from torch.utils.data import Dataset, DataLoader

# my imports
import sys
sys.path.append('../src/')
from ConvNet import ConvNet
from GravityWavesDataset import GravityWavesDataset

import argparse

# Get run directory from argparser
parser = argparse.ArgumentParser(description='Train ConvNet')
parser.add_argument('--transform', metavar='transform', type=str, nargs='+',
                                                   help='transform used, either minmax standard or none')
parser.add_argument('--init_epoch', metavar='init_epoch', type=int, nargs='+',
                                                   help='epoch to start from, if 0, start from scratch')
parser.add_argument('--model_name', metavar='model_name', type=str, nargs='+',
                                                   help='name modelfor saving')
parser.add_argument('--seed', metavar='seed', type=int, nargs='+', default=1,
                              help='initialization seed, only needed if starting from scratch and epoch=0')
parser.add_argument('--n_out', metavar='n_out', type=int, nargs='+', default=33,
                                      help='number of levels to predict, default 33. Can be up to 40')



## TODO: add arguments for batch size, learning rate, n_epochs

args = parser.parse_args()
transform = args.transform[0]
init_epoch = args.init_epoch[0]
model_name = args.model_name[0]
seed = args.seed[0]
n_out = args.n_out[0]
print(f"Training convnet with {transform} scaler, start from epoch {init_epoch}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set up directories and files
data_dir = "/scratch/users/lauraman/MiMA/runs/train_wavenet/"
filename = "atmos_daily_0.nc"

# Transform can be minmax or standard or none

if transform == "standard":
    print("Using standard scaler")
    means_filename = "atmos_daily_0_mean.nc"
    sd_filename = "atmos_daily_0_std.nc"
    transform_dict = {"filename_mean":means_filename, 
                      "filename_sd":sd_filename}
elif transform == "minmax":
    print("Using min-max scaler")
    min_filename = "atmos_daily_0_min.nc"
    max_filename = "atmos_daily_0_max.nc"
    transform_dict = {"filename_min":min_filename, 
                      "filename_max":max_filename}

gw_dataset = GravityWavesDataset(data_dir, filename, npfull_out=n_out,
                                 load=True, 
                                 transform = transform,
                                 transform_dict = transform_dict)

# Validation set
valid_filename = "atmos_daily_1.nc"
valid_dataset = GravityWavesDataset(data_dir, valid_filename, npfull_out=n_out,
                                 load=True,
                                 transform = transform,
                                 transform_dict = transform_dict)

# Save directories
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/{model_name}/"
os.makedirs(save_dir, exist_ok=True)
model_filename = f"convnet_model.pth"
path_to_model = f"{save_dir}{model_filename}"
weights_filename = f"convnet_weights.pth"
path_to_weights = f"{save_dir}{weights_filename}"
losses_filename = f"convnet_losses.csv"
path_to_losses = f"{save_dir}{losses_filename}"
training_losses_filename = f"convnet_training_losses.csv"
path_to_training_losses = f"{save_dir}{training_losses_filename}"
validation_losses_filename = f"convnet_validation_losses.csv"
path_to_validation_losses = f"{save_dir}{validation_losses_filename}"

# Set batch size 
batch_size = 2048
n_samples = len(gw_dataset)
n_batches = n_samples//batch_size

# Set up dataloaders
train_dataloader = DataLoader(gw_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=8)

# Set up model
if init_epoch==0:
    torch.manual_seed(seed)
    my_model = ConvNet(n_in=40, n_out=n_out)
    losses =[]
    training_losses = []
    validation_losses = []
    torch.save(my_model, path_to_model)

else:
    # Load model
    # Save model at end of each epoch
    weights_filename = f"convnet_weights_epoch{init_epoch}.pth"
    path_to_weights = f"{save_dir}{weights_filename}"

    my_model = torch.load(path_to_model)
    model_weights = torch.load(path_to_weights)
    my_model.load_state_dict(model_weights)

    losses = np.loadtxt(path_to_losses).tolist()
    training_losses = np.loadtxt(path_to_training_losses).tolist()
    validation_losses = np.loadtxt(path_to_validation_losses).tolist()


my_model = my_model.to(device)


n_valid = 2**20  # select only first 3rd of the year for validation, reserve rest for testing
n_epochs = 10
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)
loss_func = MSELoss()

print(f"Running {n_epochs} epochs with {n_batches} batchs of batch size={batch_size} \
for dataset size {n_samples}")

### TRAINING LOOP ###
for ep in range(init_epoch+1, n_epochs):
    i = 0
    training_loss = 0
    my_model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        X, Y = batch["X"], batch["Y"].squeeze()
        u = X[...,:40]
        T = X[...,40:80]
        X = torch.cat((u, T), dim=1)
        X, Y = X.to(device), Y.to(device)
        Y_pred = my_model(X)
        err = loss_func(Y_pred, Y)
        err.backward()
        optimizer.step()
        losses.append(err.item())
        training_loss += err.item()
        if i % 100 == 0:
            print(f"iteration: {i}, Loss:{err.item()}")
        i+=1

    training_loss = training_loss / i
    
    print(f"Training done for epoch {ep}. Validation...")

    ## Validation loss
    i = 0
    valid_loss = 0
    my_model.eval()
    for batch in valid_dataloader:
        X, Y = batch["X"], batch["Y"].squeeze()
        u = X[...,:40]
        T = X[...,40:80]
        X = torch.cat((u, T), dim=1)
        X, Y = X.to(device), Y.to(device)
        Y_pred = my_model(X)
        err = loss_func(Y_pred, Y)
        valid_loss += err.item()
        i+=1
        if i >= n_valid:
            print("validation done")
            break
        
    valid_loss = valid_loss / i
    
    print(f"Epoch: {ep}, loss:{losses[-1]}, \
          total training loss: {training_loss} \
          total validation loss: {valid_loss}")
    
    training_losses.append(training_loss)
    validation_losses.append(valid_loss)

    # Save model at end of each epoch
    weights_filename = f"convnet_weights_epoch{ep}.pth"
    path_to_weights = f"{save_dir}{weights_filename}"
    torch.save(my_model.state_dict(), path_to_weights)
    # Save losses
    np.savetxt(path_to_losses, np.array(losses), delimiter=",")
    np.savetxt(path_to_training_losses, np.array(training_losses), delimiter=",")
    np.savetxt(path_to_validation_losses, np.array(validation_losses), delimiter=",")

print(f"Done training {ep} epochs")

# Save final version
weights_filename = f"convnet_weights.pth"
path_to_weights = f"{save_dir}{weights_filename}"
torch.save(my_model.state_dict(), path_to_weights)


