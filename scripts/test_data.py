import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('../src/')
from GravityWavesDataset import GravityWavesDataset

import argparse

# This script saves test data into same format as we will save the test predictions for each model.
# It saves X, Y and X_scaled, Y_scaled
# Only need to run this once

# Get run directory from argparser
parser = argparse.ArgumentParser(description='Save test data for wavenet in same format as test data for each model')
parser.add_argument('--transform', metavar='transform', type=str, default=None,
                                                   help='transform used, either minmax standard or none')
parser.add_argument('--subset_time', metavar='subset_time', type=int, nargs='+', default=None,
                                              help='subset of data to use. Either None or tuple as x1 x2 \
                                                      e.g. (150, 240) for JJA. Currently only contininous time slicing \
                                                      can be implemented. Will allow (-30,60) for DJF')
parser.add_argument('--train_dir', metavar='train_dir', type=str, default="/scratch/users/lauraman/MiMA/runs/train_wavenet/",
                help='directory for training data, required for scaling appropriately')
parser.add_argument('--train_filename', metavar='train_filename', type=str, default="atmos_daily_0",
        help='filename for training data, required for scaling appropriately. should be either atmos_daily_0 or atmos_all_12')
parser.add_argument('--test_dir', metavar='test_dir', type=str, default="/scratch/users/lauraman/MiMA/runs/train_wavenet/",
                help='directory for test data')
parser.add_argument('--test_filename', metavar='test_filename', type=str, default="atmos_daily_2",
        help='filename for testing')
parser.add_argument('--save_filename', metavar='save_filename', type=str, default="atmos_daily_2",
        help='filename for saving')

args = parser.parse_args()
print(args)
transform = args.transform
subset_time = args.subset_time
if subset_time != None:
    subset_time = tuple(subset_time)
train_dir = args.train_dir
test_dir = args.test_dir
train_filestart = args.train_filename
test_filestart = args.test_filename
save_filestart = args.save_filename

device = "cpu"

# Set up directories and files
transform_dir = train_dir

if transform == "standard":
    print("Using standard scaler")
    means_filename = f"{train_filestart}_mean.nc"
    sd_filename = f"{train_filestart}_std.nc"
    transform_dict = {"filename_mean":means_filename, 
                      "filename_sd":sd_filename}
elif transform == "minmax":
    print("Using min-max scaler")
    min_filename = f"{train_filestart}_min.nc"
    max_filename = f"{train_filestart}_max.nc"
    transform_dict = {"filename_min":min_filename, 
                      "filename_max":max_filename}

transform_dict["transform_dir"] = transform_dir
print(f"Using transform from {transform_dir}/{train_filestart}.nc")
print(f"Test file is {test_dir}/{test_filestart}.nc")

print("Loading test dataset into memory.")

# Test set
np_out = 40
test_filename = f"{test_filestart}.nc"
test_dataset = GravityWavesDataset(test_dir, test_filename,
                                   npfull_out=np_out,
                                   transform = transform,
                                   transform_dict = transform_dict)
print("Data loaded.")

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
                              num_workers=16, shuffle=False)


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

# Set up DataArrays to save: X, X_scaled, Y_truth, Y_truth_scaled

X_save = xr.DataArray(np.zeros((ntime, npfull_in, nlat, nlon)), 
                                coords={ 'time':time,
                                         'pfull_in':pfull_in,
                                         'lat':lat,
                                         'lon':lon },
                                dims=['time', 'pfull_in', 'lat', 'lon'] ,
                                name="X" )

X_scaled_save = xr.DataArray(np.zeros((ntime, npfull_in, nlat, nlon)), 
                                coords={ 'time':time,
                                         'pfull_in':pfull_in,
                                         'lat':lat,
                                         'lon':lon },
                                dims=['time', 'pfull_in', 'lat', 'lon'] ,
                                name="X_scaled" )

Y_truth_save = xr.DataArray(np.zeros((ntime, npfull_out, nlat, nlon)), 
                                coords={ 'time':time,
                                         'pfull':pfull,
                                         'lat':lat,
                                         'lon':lon },
                                dims=['time', 'pfull', 'lat', 'lon'] ,
                                name="Y_truth" )

Y_truth_scaled_save = xr.DataArray(np.zeros((ntime, npfull_out, nlat, nlon)), 
                                coords={ 'time':time,
                                         'pfull':pfull,
                                         'lat':lat,
                                         'lon':lon },
                                dims=['time', 'pfull', 'lat', 'lon'] ,
                                name="Y_truth_scaled" )

# File will be saved in 
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/truth/"
## Save all X,Y and X_scaled, Y_scaled
print("Arrays set up. Open all points in test dataset")
time_ind = 0
lat_ind = 0
## fill all lons at once, with batch size = nlon
for batch in test_dataloader:
        X_scaled, Y_scaled = batch["X"].squeeze().to('cpu'), batch["Y"].squeeze().to('cpu')

        # Save scaled variables, after reshaping into correct size
        Y_truth_scaled_save[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y_scaled.T.reshape(npfull_out, nlat_per_batch, nlon)
        X_scaled_save[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = X_scaled.T.reshape(npfull_in, nlat_per_batch, nlon)
        
        # Inverse scaler
        X, Y = test_dataset.inverse_scaler(X_scaled, Y_scaled)
        Y_truth_save[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y.T.reshape(npfull_out, nlat_per_batch, nlon)
        X_save[time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = X.T.reshape(npfull_in, nlat_per_batch, nlon)

        lat_ind += nlat_per_batch

        if lat_ind == nlat:
            # increment to next timestep and reset lat
            time_ind += 1
            lat_ind = 0
        
        
        if time_ind % 30 == 0:
            print(f"Calculated up to time={time_ind}")
            save_as = f"{save_dir}/{save_filestart}_tmp_{time_ind}.nc"
            ds = X_scaled_save.to_dataset(name = "X_scaled")
            ds["X"] = X_save
            ds["Y_truth_scaled"] = Y_truth_scaled_save
            ds["Y_truth"] = Y_truth_save
            ds.to_netcdf(save_as)
            print(f"Saved as {save_as}... continuing")
        
        
print("Done. Saving to netcdf")

save_as = f"{save_dir}/{save_filestart}_truth.nc"
ds = X_scaled_save.to_dataset(name = "X_scaled")
ds["X"] = X_save
ds["Y_truth_scaled"] = Y_truth_scaled_save
ds["Y_truth"] = Y_truth_save
ds.to_netcdf(save_as)
print(f"Done. Saved as {save_as}")



