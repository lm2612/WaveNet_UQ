import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib

from torch.nn import MSELoss, Module
from torch.utils.data import Dataset, DataLoader

# my imports
import sys
sys.path.append('../src/')
from Wavenet import Wavenet
from GravityWavesDataset import GravityWavesDataset
from utils import get_best_epoch, check_nc

import argparse

# Get arguments from argparser
parser = argparse.ArgumentParser(description='Test wavenet offline')

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
        help='filename for testing data, can be multiple files. \
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
scaler_filestart = args.scaler_filestart

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set up directories and files
model_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/{model_name}/"   # models saved here
transform_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"     # standard scalar files saved here
test_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"          # test data saved here
if transform == "standard":
    print("Using standard scaler")
    means_filename = f"{scaler_filestart}_mean.nc"
    sd_filename = f"{scaler_filestart}_std.nc"
    transform_dict = {"filename_mean":means_filename,
                      "filename_sd":sd_filename}
elif transform == "minmax":
    print("Using min-max scaler")
    min_filename = f"{scaler_filestart}_min.nc"
    max_filename = f"{scaler_filestart}_max.nc"
    transform_dict = {"filename_min":min_filename,
                      "filename_max":max_filename}

transform_dict["transform_dir"] = transform_dir
print(f"Using transform from {transform_dir}/{scaler_filestart}*.nc")
print(f"Test data is in {test_dir}. {filename}")

print("Loading test dataset into memory.")

# Test set
np_out = 40
test_dataset = GravityWavesDataset(test_dir, filename,
                                   npfull_out=np_out,
                                   subset_time=subset_time,
                                   transform = transform,
                                   transform_dict = transform_dict,
                                   component = component)
print("Data loaded.")


# Get model with lowest validation error 
epoch = get_best_epoch(model_dir, min_epochs=50)
print(f"Model with lowest validation error is epoch {epoch}")

model_filename = f"wavenet_model.pth"
path_to_model = f"{model_dir}{model_filename}"
weights_filename = f"wavenet_weights_epoch{epoch}.pth"
path_to_weights = f"{model_dir}{weights_filename}"

print(f"Loading model and weights: {path_to_model}, {path_to_weights}")
my_model = torch.load(path_to_model,  map_location = device)
model_weights = torch.load(path_to_weights,  map_location = device)

my_model.load_state_dict(model_weights)
my_model.to(device)
my_model.eval()

# Set up dataloader, batch size = nlon = 128 so we can fill data arrays easily
ntime, npfull_out, nlat, nlon = test_dataset.ntime, test_dataset.npfull_out, test_dataset.nlat, test_dataset.nlon
print(ntime, npfull_out, nlat, nlon)

nlat_per_batch = 64
batch_size = nlon * nlat_per_batch
nlat_splits = nlat // nlat_per_batch
print(nlat_per_batch, batch_size)
n_samples = len(test_dataset)
n_batches = n_samples//batch_size

# Set up dataloader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                              num_workers=4, shuffle=False)


## Save arrays into xarray Dataset
# Set up dims
time = test_dataset.time
pfull = test_dataset.pfull
lat = test_dataset.lat
lon = test_dataset.lon
pfull_in = np.concatenate( (pfull, 
                            pfull, 
                            np.array([np.nan, 
                                      np.nan])) )
npfull_in = len(pfull_in)

## Create empty data arrays to save predictions to
# Get variables 
if component.lower() == "zonal":
    gwf_comp = "gwfu_cgwd"
elif component.lower() == "meridional":
    gwf_comp = "gwfv_cgwd"
gwfu_pred = xr.zeros_like(test_dataset.ds[gwf_comp])
gwfu_pred.name = f"{gwf_comp}_pred"
gwfu_pred_scaled = xr.zeros_like(test_dataset.ds[gwf_comp])
gwfu_pred_scaled.name =  f"{gwf_comp}_pred_scaled"
                                
## Make predictions for all timesteps in test_dataset [1 yr of data]
print("Arrays set up. Predicting on all points in test dataset")
time_ind = 0
lat_ind = 0
## fill all lons at once, with batch size = nlon
for batch in test_dataloader:
        X_scaled, Y_scaled = batch["X"].squeeze().to(device), batch["Y"].squeeze().to('cpu')
        Y_pred_scaled = my_model(X_scaled).squeeze().detach().to('cpu')
        X_scaled = X_scaled.to('cpu')

        # Reshape
        Y_pred_reshaped = Y_pred_scaled.T.reshape(npfull_out, nlat_per_batch, nlon)
        # Save scaled variable for future
        gwfu_pred_scaled[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y_pred_reshaped
        # Unscale and save unscaled variable
        Y_pred = test_dataset.inverse_standard_scaler(Y_pred_reshaped, 
                                           test_dataset.gwfu_mean[0, :, lat_ind:lat_ind+nlat_per_batch], 
                                           test_dataset.gwfu_sd[0, :, lat_ind:lat_ind+nlat_per_batch])
        gwfu_pred[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y_pred 

        lat_ind += nlat_per_batch
        
        if lat_ind == nlat:
            # increment to next timestep and reset lat
            time_ind += 1
            lat_ind = 0
        
        
        
print("All predictions done. Saving to netcdf")
## What to save as
if len(args.filename)==1:
    save_filestart = filename.removesuffix(".nc")
else:
    save_filestart = filename[0].removesuffix(".nc")
    year_indices = [file.removesuffix(".nc").split("_")[-1] for file in filename]
    save_filestart = f"{save_filestart}-{year_indices[-1]}"


save_as = f"{model_dir}/{save_filestart}.nc"
ds = gwfu_pred.to_dataset(name = f"{gwf_comp}_pred")
ds[f"{gwf_comp}_pred_scaled"] = gwfu_pred_scaled
ds.to_netcdf(save_as)
print(f"Done. Saved as {save_as}")



