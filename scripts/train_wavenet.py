import os 
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random

from torch.nn import MSELoss, Module
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# my imports
import sys
sys.path.append('../src/')
from utils import check_nc
from Wavenet import Wavenet
from GravityWavesDataset import GravityWavesDataset
from custom_loss import neg_log_likelihood_loss

import argparse

# Get arguments from argparser
parser = argparse.ArgumentParser(description='Train wavenet')

## Arguments that define the model/data
parser.add_argument('--component', metavar='component', type=str, 
        default="zonal", 
        help='directional component of gwd to predict, either zonal \
                or meridional. Default is zonal')
parser.add_argument('--aleatoric', action=argparse.BooleanOptionalAction,
        help='turns on aleatoric uncertainty, meaning model produces mean \
                 and std. Loss function is log likelihood rather than MSE')
parser.add_argument('--transform', metavar='transform', type=str, 
        default=None,
        help='transform used, either minmax standard or none')
parser.add_argument('--n_out', metavar='n_out', type=int, default=40,
        help='number of levels to predict up to 40 (max 40). e.g. 33 to \
                ignore zero levels')
parser.add_argument('--subset_time', metavar='subset_time', type=int,
        nargs='+', default=None,
        help='subset of data to use. Either None or tuple as x1 x2 \
                e.g. (150, 240) for JJA. Currently only contininous \
                time slicing can be implemented. Will allow (-30,60) for DJF')

## File names for training, valid, scaling and saving
parser.add_argument('--model_name', metavar='model_name', type=str, 
        help='name of model for saving (used to create new dir)')
parser.add_argument('--filename', metavar='filename', type=str,
        nargs='+', default="atmos_all_12",
        help='filename for training data, can be multiple files. \
                File suffix should be .nc and will be added if not present here')
parser.add_argument('--scaler_filestart', metavar='scaler_filestart', 
        type=str, default="atmos_all_12",
        help='start of filename for files containing means and std \
                for scaling:  should be consistent with training data \
                e.g. atmos_all_12')
parser.add_argument('--valid_filename', metavar='valid_filename',
        type=str, default="atmos_daily_11",
        help='filename for validation data, e.g. atmos_all_13.\
                File suffix should be .nc and will be added if not present here')

## Arguments for training settings
parser.add_argument('--seed', metavar='seed', type=int,  default=1,
        help='initialization seed, only needed if starting from scratch \
                and epoch=0')
parser.add_argument('--init_epoch', metavar='init_epoch', type=int,
        default=0,
        help='epoch to start from, if 0, start from scratch')
parser.add_argument('--n_epoch', metavar='n_epoch', type=int,
        default=10,
        help='number of epochs to train for')
parser.add_argument('--dropout_rate', metavar='dropout_rate',
        type=float, default=0,
        help='dropout rate: a floating point number between 0 and 1. \
                0 means no dropout')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, 
        default=1e-4,
        help='learning_rate, small number e.g. 1e-4')
parser.add_argument('--weight_decay', metavar='weight_decay', type=float, 
        default=0.,
        help='weight decay, e.g. 1e-4')

## Set up args
args = parser.parse_args()
print(args)
# Model arguments
component = args.component
aleatoric = args.aleatoric
transform = args.transform
n_out = args.n_out
subset_time = args.subset_time
if subset_time != None:
    subset_time = tuple(subset_time)

# Filename arguments
model_name = args.model_name
if model_name == None:
    print("Model name not provided, expect errors - will not be able to save output.")
if len(args.filename)==1:
    filename = check_nc(args.filename[0])
else:
    filename = [check_nc(file_i) for file_i in args.filename]
filestart = args.scaler_filestart
valid_filename = check_nc(args.valid_filename)

# Training arguments
seed = args.seed
init_epoch = args.init_epoch
n_epoch = args.n_epoch
dropout_rate = args.dropout_rate
learning_rate = args.learning_rate
weight_decay = args.weight_decay

# Key info
print(f"Training {component} wavenet with {transform} scaler, start from epoch {init_epoch}. \
        Seed = {seed}. Using dropout? {dropout_rate}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set up directories and files
data_dir = "/scratch/users/lauraman/MiMA/runs/train_wavenet/"

print(f"Training file(s): {filename}")
# Transform can be minmax or standard or none
if transform == "standard":
    print("Using standard scaler")
    means_filename = f"{filestart}_mean.nc"
    sd_filename = f"{filestart}_std.nc"
    transform_dict = {"filename_mean":means_filename, 
                      "filename_sd":sd_filename}
elif transform == "minmax":
    print("Using min-max scaler")
    min_filename = f"{filestart}_min.nc"
    max_filename = f"{filestart}_max.nc"
    transform_dict = {"filename_min":min_filename, 
                      "filename_max":max_filename}

gw_dataset = GravityWavesDataset(data_dir, filename,
                                 npfull_out=n_out,
                                 subset_time=subset_time,
                                 transform = transform,
                                 transform_dict = transform_dict,
                                 component = component)

# Validation set
print(f"Validation file: {valid_filename}")
valid_dataset = GravityWavesDataset(data_dir, valid_filename,
                                    npfull_out=n_out,
                                    subset_time=subset_time,
                                    transform = transform,
                                    transform_dict = transform_dict, 
                                    component = component)

# Save directories
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/{model_name}/"
os.makedirs(save_dir, exist_ok=True)
model_filename = f"wavenet_model.pth"
path_to_model = f"{save_dir}{model_filename}"
weights_filename = f"wavenet_weights.pth"
path_to_weights = f"{save_dir}{weights_filename}"
losses_filename = f"wavenet_losses.csv"
path_to_losses = f"{save_dir}{losses_filename}"
training_losses_filename = f"wavenet_training_losses.csv"
path_to_training_losses = f"{save_dir}{training_losses_filename}"
validation_losses_filename = f"wavenet_validation_losses.csv"
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

# Settings needed if aleatoric
if aleatoric:
    n_d=[128,64,32,2]
    loss_func = neg_log_likelihood_loss      # custom loss function 
    print("Aleatoric model, predicting mean and std with neg log likelihood loss")
 
else:
    n_d=[128,64,32,1]
    loss_func = MSELoss()
    print("Regular model, predicting output only with MSELoss")

# Set up model
if init_epoch==0:
    torch.manual_seed(seed)
    my_model = Wavenet(n_d=n_d, n_in=82, n_out=n_out, dropout_rate=dropout_rate)
    losses =[]
    training_losses = []
    validation_losses = []
    torch.save(my_model, path_to_model)

else:
    # Load model
    # Save model at end of each epoch
    weights_filename = f"wavenet_weights_epoch{init_epoch}.pth"
    path_to_weights = f"{save_dir}{weights_filename}"

    my_model = torch.load(path_to_model)
    model_weights = torch.load(path_to_weights)
    my_model.load_state_dict(model_weights)

    losses = np.loadtxt(path_to_losses).tolist()
    training_losses = np.loadtxt(path_to_training_losses).tolist()
    validation_losses = np.loadtxt(path_to_validation_losses).tolist()


my_model = my_model.to(device)

# Set up optimizer and loss function
optimizer = torch.optim.Adam(my_model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay)
## Linearly decrease learning rate over 200 epochs from 1e-3 to 1e-5
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
        end_factor=0.01, total_iters=200)

print(f"Running {n_epoch} epochs with {n_batches} batchs of batch size={batch_size} \
        for dataset size {n_samples}")

### TRAINING LOOP ###
for ep in range(init_epoch+1, n_epoch):
    i = 0
    training_loss = 0
    my_model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        X, Y = batch["X"].squeeze(), batch["Y"].squeeze()
        X, Y = X.to(device), Y.to(device)
        Y_pred = my_model(X).squeeze()
        err = loss_func(Y_pred, Y)
        err.backward()
        optimizer.step()
        losses.append(err.item())
        training_loss += err.item()
        if i % 1000 == 0:
            print(f"iteration: {i}, Loss:{err.item()}")
        i+=1

    training_loss = training_loss / i
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: ADAM lr %.4f -> %.4f" % (ep, before_lr, after_lr))
    print(f"Training done for epoch {ep}. Validation...")
    

    ## Validation loss
    i = 0
    valid_loss = 0
    my_model.eval()
    for batch in valid_dataloader:
        X, Y = batch["X"].squeeze(), batch["Y"].squeeze()
        X, Y = X.to(device), Y.to(device)
        Y_pred = my_model(X).squeeze()
        err = loss_func(Y_pred, Y)
        valid_loss += err.item()
        i+=1
        
    valid_loss = valid_loss / i
    
    print(f"Epoch: {ep}, loss:{losses[-1]}, \
          total training loss: {training_loss} \
          total validation loss: {valid_loss}")
    
    training_losses.append(training_loss)
    validation_losses.append(valid_loss)

    # Save model at end of each epoch
    weights_filename = f"wavenet_weights_epoch{ep}.pth"
    path_to_weights = f"{save_dir}{weights_filename}"
    torch.save(my_model.state_dict(), path_to_weights)
    # Save losses
    np.savetxt(path_to_losses, np.array(losses), delimiter=",")
    np.savetxt(path_to_training_losses, np.array(training_losses), delimiter=",")
    np.savetxt(path_to_validation_losses, np.array(validation_losses), delimiter=",")

print(f"Done training {ep} epochs")

# Save final version
weights_filename = f"wavenet_weights.pth"
path_to_weights = f"{save_dir}{weights_filename}"
torch.save(my_model.state_dict(), path_to_weights)


