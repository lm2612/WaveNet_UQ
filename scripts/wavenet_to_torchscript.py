import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import random

from torch.nn import MSELoss, Module
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append('../src/')
from utils import *
from Wavenet_for_MiMA import Wavenet_for_MiMA
from GravityWavesDataset import GravityWavesDataset 
import argparse

# Get arguments from argparser
parser = argparse.ArgumentParser(description='Save wavenet pytorch model to torchscript')

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

## File names for training, valid, scaling and saving
parser.add_argument('--model_name', metavar='model_name', type=str,
        help='name of model for saving (used to create new dir)')
parser.add_argument('--filename', metavar='filename', type=str,
        nargs='+', default="atmos_all_11",
        help='filename for testing the model runs in torchscript (relatively unimportant) \
                File suffix should be .nc and will be added if not present here')
parser.add_argument('--scaler_filestart', metavar='scaler_filestart',
        type=str, default="atmos_all_12",
        help='start of filename for files containing means and std \
                for scaling:  should be consistent with training data \
                e.g. atmos_all_12')


## Set up args
args = parser.parse_args()
print(args)
# Model arguments
component = args.component
aleatoric = args.aleatoric
transform = args.transform
n_out = args.n_out

# Filename arguments
model_name = args.model_name

if len(args.filename)==1:
    filename = check_nc(args.filename[0])
else:
    filename = [check_nc(file_i) for file_i in args.filename]
filestart = args.scaler_filestart



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
print(device)

#data_dir = "/scratch/users/lauraman/MiMA/runs/train_wavenet/"
data_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"

np_out = 40

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

# Test set, supposed to be unseen to model but this is unimportant and is purely for checking
# array sizes. We only take first 20 timesteps.
gw_dataset = GravityWavesDataset(data_dir, filename,
                                 npfull_out =n_out,
                                 subset_time = (0,20),
                                 transform = transform,
                                 transform_dict = transform_dict,
                                 component = component)
# Also open raw dataset with no transform
gw_dataset_raw = GravityWavesDataset(data_dir, filename,
                                 npfull_out =n_out,
                                 subset_time = (0,20),
                                 component = component)

# Set batch size: 128 for all lons at once 
batch_size = 128
n_samples = len(gw_dataset)
n_batches = n_samples//batch_size

# Set up dataloader for testing
gw_dataloader = DataLoader(gw_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
gw_dataloader_raw = DataLoader(gw_dataset_raw, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

# Get transform info to save into model
transform_vars = {"gwfu_mean": torch.tensor(gw_dataset.gwfu_mean),
                  "u_mean": torch.tensor(gw_dataset.u_mean),
                  "T_mean": torch.tensor(gw_dataset.T_mean),
                  "ps_mean": torch.tensor(gw_dataset.ps_mean),
                  "gwfu_sd": torch.tensor(gw_dataset.gwfu_sd),
                  "u_sd": torch.tensor(gw_dataset.u_sd),
                  "T_sd": torch.tensor(gw_dataset.T_sd),
                  "ps_sd": torch.tensor(gw_dataset.ps_sd) }

## Load model weights
model_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/{model_name}/"
epoch = get_best_epoch(model_dir)
print(f"Model with lowest validation error is epoch {epoch}")

weights_filename = f"wavenet_weights_epoch{epoch}.pth"
path_to_weights = f"{model_dir}{weights_filename}"
model_weights = torch.load(path_to_weights, map_location = device)

## New instance of model with these weights
model_for_mima = Wavenet_for_MiMA(n_in=82, n_out=40, transform_vars = transform_vars)
model_for_mima.load_state_dict(model_weights)
model_for_mima.double()
model_for_mima.eval()


## Test model works
batch = next(iter(gw_dataloader_raw))
X_raw, Y_raw = batch["X"].squeeze(), batch["Y"].squeeze() 
X_raw = X_raw.double()
j = 0
# mima model - pass in raw u,T,lat,ps and lat ind, scaling done internally
u = X_raw[:, :40]
T = X_raw[:, 40:80]
lat = X_raw[:, 80:81]*np.pi/180.
ps = X_raw[:, 81:82]
lat_ind = j*torch.ones(batch_size).reshape(batch_size, 1)

print("Test python version of model for single lat ind")
Y_pred_mima = model_for_mima(u, T, lat, ps, lat_ind)
print("Success")
print(Y_pred_mima)

## Try two indices at once, as needed
batch = next(iter(gw_dataloader_raw))
X_raw, Y_raw = batch["X"].squeeze(), batch["Y"].squeeze()
X_raw = X_raw.double()
j = 1
# mima model - pass in raw u,T,lat,ps and lat ind, scaling done internally
u = torch.concat((u, X_raw[:, :40]), dim=0)
T = torch.concat((T, X_raw[:, 40:80]),  dim=0)
lat = torch.concat((lat, X_raw[:, 80:81]*np.pi/180.), dim=0)
ps = torch.concat((ps, X_raw[:, 81:82]), dim=0)
lat_ind = torch.concat((lat_ind, j*torch.ones(batch_size).reshape(batch_size, 1) ), dim=0)

print("Test python version again with multiple lat inds")
Y_pred_mima = model_for_mima(u, T, lat, ps, lat_ind)
print("Success")
print(Y_pred_mima)

## Save model as torchscript
print("Exporting model to torchscript")
traced_model = torch.jit.trace(model_for_mima, example_inputs = (u, T, lat, ps, lat_ind ) )
frozen_model = torch.jit.freeze(traced_model)

## Test as traced model
print("Testing traced model several times")
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
print("Done")

filename = f"{model_dir}/{component}_wavenet.pth"

frozen_model.save(filename)
rint(f"Saved to {filename}")
