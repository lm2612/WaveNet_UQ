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
from utils import count_parameters
from utils import init_xavier 
from Wavenet_for_MiMA import Wavenet_for_MiMA
from GravityWavesDataset import GravityWavesDataset 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
print(device)


data_dir = "/scratch/users/lauraman/MiMA/runs/train_wavenet/"

np_out = 40
# Transform can be minmax or standard
transform = "standard"
startfile="atmos_all_12-15"
means_filename = f"{startfile}_mean.nc"
sd_filename = f"{startfile}_std.nc"
transform_dict = {"filename_mean":means_filename, 
                  "filename_sd":sd_filename}
ds_mean = xr.open_dataset(data_dir + means_filename, decode_times=False )
ds_sd = xr.open_dataset(data_dir + sd_filename, decode_times=False )
gwfu_mean = ds_mean["gwfu_cgwd"] #.mean(axis=(2, 3))
gwfu_sd = ds_sd["gwfu_cgwd"]

# Test set - entirely unseen to model (valid dataset is atmos_daily_11)
valid_filename = "atmos_daily_10.nc"
subset_time=(0,90)
valid_dataset = GravityWavesDataset(data_dir, valid_filename, npfull_out=np_out,
                                 transform = transform,
                                 transform_dict = transform_dict, 
                                    subset_time=(0,90))
valid_dataset_raw =  GravityWavesDataset(data_dir, valid_filename, npfull_out=np_out, 
                                         subset_time=(0,90))
path_to_file = data_dir + valid_filename
ds = xr.open_dataset(path_to_file, decode_times=False )
# Get dimensions
lon = ds["lon"]
lat = ds["lat"]
time = ds["time"]
pfull = ds["pfull"]
ps = ds["ps"]
gwfu = ds["gwfu_cgwd"]
ucomp = ds["ucomp"]


print("Done.")
lat = lat.to_numpy()
print(lat)
lat = lat*np.pi/180.
print(lat)

# Set batch size 
batch_size = 128
n_samples = len(valid_dataset)
n_batches = n_samples//batch_size

# Set up dataloader for testing
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
valid_dataloader_raw = DataLoader(valid_dataset_raw, batch_size=batch_size,
                              num_workers=0)


transform_vars = {"gwfu_mean": torch.tensor(valid_dataset.gwfu_mean),
                  "u_mean": torch.tensor(valid_dataset.u_mean),
                  "T_mean": torch.tensor(valid_dataset.T_mean),
                  "ps_mean": torch.tensor(valid_dataset.ps_mean),
                  "gwfu_sd": torch.tensor(valid_dataset.gwfu_sd),
                  "u_sd": torch.tensor(valid_dataset.u_sd),
                  "T_sd": torch.tensor(valid_dataset.T_sd),
                  "ps_sd": torch.tensor(valid_dataset.ps_sd) }

## Load model weights
component="meridional" # meridional or zonal
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/wavenet_4xdaily_4yr_scaling_{component}_standard_seed1/"
epoch=43
weights_filename = f"wavenet_weights_epoch{epoch}.pth"
path_to_weights = f"{save_dir}{weights_filename}"
model_weights = torch.load(path_to_weights, map_location = device)

## New instance of model with these weights
model_for_mima = Wavenet_for_MiMA(n_in=82, n_out=40, transform_vars = transform_vars)
model_for_mima.load_state_dict(model_weights)
model_for_mima.double()
model_for_mima.eval()


## Test model works
batch = next(iter(valid_dataloader_raw))
X_raw, Y_raw = batch["X"].squeeze(), batch["Y"].squeeze() 
X_raw = X_raw.double()
j = 0
# mima model - pass in raw u,T,lat,ps and lat ind, scaling done internally
u = X_raw[:, :40]
T = X_raw[:, 40:80]
lat = X_raw[:, 80:81]
ps = X_raw[:, 81:82]

print(lat.shape)
lat_ind = j*torch.ones(batch_size).reshape(batch_size, 1)

#print(lat)
#print(lat_ind)

print("test python")
Y_pred_mima = model_for_mima(u, T, lat, ps, lat_ind)
print("success")

## Try two indices at once, as needed
batch = next(iter(valid_dataloader_raw))
X_raw, Y_raw = batch["X"].squeeze(), batch["Y"].squeeze()
X_raw = X_raw.double()
j = 1
# mima model - pass in raw u,T,lat,ps and lat ind, scaling done internally
u = torch.concat((u, X_raw[:, :40]), dim=0)
T = torch.concat((T, X_raw[:, 40:80]),  dim=0)
lat = torch.concat((lat, X_raw[:, 80:81]), dim=0)
ps = torch.concat((ps, X_raw[:, 81:82]), dim=0)

lat_ind = torch.concat((lat_ind, j*torch.ones(batch_size).reshape(batch_size, 1) ), dim=0)

print(lat)
print(lat_ind)
print("test python, x2 lat inds")
Y_pred_mima = model_for_mima(u, T, lat, ps, lat_ind)
print("success")

## Save model as torchscript
traced_model = torch.jit.trace(model_for_mima, example_inputs = (u, T, lat, ps, lat_ind ) )
frozen_model = torch.jit.freeze(traced_model)

## Test as traced model
print("Test traced model")
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
Y_pred_mima = traced_model(u, T, lat, ps, lat_ind)
print("Done")

filename = f"{save_dir}/saved_{component}_wavenet.pth"

frozen_model.save(filename)
print(f"Saved to {filename}")
