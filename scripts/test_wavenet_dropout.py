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
from utils import enable_dropout

import argparse

# Get run directory from argparser
parser = argparse.ArgumentParser(description='Test wavenet')
parser.add_argument('--transform', metavar='transform', type=str, nargs='+',
                                                           help='transform used, either minmax standard or none')
parser.add_argument('--model_name', metavar='model_name', type=str, nargs='+',
                                                           help='name modelfor saving')
parser.add_argument('--epoch', metavar='epoch', type=str, nargs='+',
                                                           help='epoch to use')



args = parser.parse_args()
transform = args.transform[0]
model_name = args.model_name[0]
epoch = args.epoch[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_dir = f"/scratch/users/lauraman/WaveNetPyTorch/models/{model_name}/"

# Set up directories and files
data_dir = "/scratch/users/lauraman/MiMA/runs/train_wavenet/"

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

print("Loading test dataset into memory.")

# Test set
np_out = 40
test_filename = "atmos_daily_2.nc"
test_dataset = GravityWavesDataset(data_dir, test_filename,
                                   npfull_out=np_out,
                                   transform = transform,
                                   transform_dict = transform_dict)
print("Data loaded.")

model_filename = f"wavenet_model.pth"
path_to_model = f"{model_dir}{model_filename}"
weights_filename = f"wavenet_weights_epoch{epoch}.pth"
path_to_weights = f"{model_dir}{weights_filename}"

print(f"Loading model and weights: {path_to_model}, {path_to_weights}")
my_model = torch.load(path_to_model)
model_weights = torch.load(path_to_weights)

my_model.load_state_dict(model_weights)
my_model.to(device)

# Keep dropout at test time, rest of model set to .eval()
enable_dropout(my_model)

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

# For dropout: sample N times
N = 100
sample = np.arange(N)

# Set up DataArrays to save
Y_pred_scaled_save = xr.DataArray(np.zeros((N, ntime, npfull_out, nlat, nlon)), 
        coords={ 'sample':sample,
                 'time':time,
                 'pfull':pfull,
                 'lat':lat,
                 'lon':lon },
        dims=['sample', 'time', 'pfull', 'lat', 'lon'] ,
        name="Y_pred_scaled" )

Y_pred_save = xr.DataArray(np.zeros((N, ntime, npfull_out, nlat, nlon)), 
        coords={ 'sample':sample, 
                 'time':time,
                 'pfull':pfull,
                 'lat':lat,
                 'lon':lon },
       dims=['sample', 'time', 'pfull', 'lat', 'lon'] ,
       name="Y_pred" )
        
## Make predictions for all timesteps in test_dataset [1 yr of data]
print("Arrays set up. Predicting on all points in test dataset")
time_ind = 0
lat_ind = 0
## fill all lons at once, with batch size = nlon
for batch in test_dataloader:
        X_scaled, Y_scaled = batch["X"].squeeze().to(device), batch["Y"].squeeze().to('cpu')
        for s in range(N):
            Y_pred_scaled = my_model(X_scaled).detach().to('cpu')

            # Save scaled variables, after reshaping into correct size
            Y_pred_scaled_save[s, time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y_pred_scaled.T.reshape(npfull_out, nlat_per_batch, nlon)
        
            # Inverse scaler
            _, Y_pred = test_dataset.inverse_scaler(X_scaled.to('cpu'), Y_pred_scaled)
            Y_pred_save[s, time_ind, :, lat_ind:lat_ind+nlat_per_batch, :] = Y_pred.T.reshape(npfull_out, nlat_per_batch, nlon)

        lat_ind += nlat_per_batch
        
        if lat_ind == nlat:
            # increment to next timestep and reset lat
            time_ind += 1
            lat_ind = 0
        
        
        if time_ind % 30 == 0:
            print(f"Calculated up to time={time_ind}")
            save_as = f"{model_dir}/test_predictions_epoch{epoch}_tmp_{time_ind}.nc"
            ds = Y_pred_scaled_save.to_dataset(name = "Y_pred_scaled")
            ds["Y_pred"] = Y_pred_save
            ds.to_netcdf(save_as)
            print(f"Saved as {save_as}... continuing")
        
        
print("All predictions done. Saving to netcdf")

save_as = f"{model_dir}/test_predictions_epoch{epoch}.nc"
ds = Y_pred_scaled_save.to_dataset(name = "Y_pred_scaled")
ds["Y_pred"] = Y_pred_save
ds.to_netcdf(save_as)
print(f"Done. Saved as {save_as}")



