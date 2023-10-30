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

## File names for training, valid, scaling and saving
parser.add_argument('--model_start', metavar='model_start', type=str,
        help='start of model name') 
parser.add_argument('--seed', metavar='seed', type=int,
        help='seed, used to find dir')
parser.add_argument('--filename', metavar='filename', type=str,
        default="atmos_all_12",
        help='filename for testing data, one file only. File suffix \
        should be .nc and will be added if not present here')
## Set up args
args = parser.parse_args()
print(args)
# Filename arguments
model_start = args.model_start
seed = args.seed

filename = check_nc(args.filename)


# Set up directories and files
ad99_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/AD99_run/"          

ds_ad99 = xr.open_dataset(f"{ad99_dir}/{filename}", decode_times=False )

# Get dimensions
lon = ds_ad99["lon"]
lat = ds_ad99["lat"]
time = ds_ad99["time"]
pfull = ds_ad99["pfull"]

ntime = len(time)
npfull = len(pfull)
nlat = len(lat)
nlon = len(lon)

gwfu_ad99 = ds_ad99["gwfu_cgwd"]
gwfv_ad99 = ds_ad99["gwfv_cgwd"]

# This model
offline_dir = "/scratch/users/lauraman/WaveNetPyTorch/models/"
online_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/"

# Take differences
offline_gwfu_diff = xr.zeros_like(gwfu_ad99)
offline_gwfv_diff = xr.zeros_like(gwfu_ad99)
online_gwfu_diff = xr.zeros_like(gwfu_ad99)
online_gwfv_diff = xr.zeros_like(gwfu_ad99)

## OFFLINE
## zonal component
component = "zonal"
ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
offline_gwfu_diff = (ds_u["gwfu_cgwd_pred"]-gwfu_ad99)
## meridional component
component = "meridional"
ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
offline_gwfv_diff = (ds_u["gwfv_cgwd_pred"]-gwfv_ad99)
## ONLINE
ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
online_gwfu_diff = (ds_u["gwfu_cgwd"]-gwfu_ad99)
online_gwfv_diff = (ds_u["gwfv_cgwd"]-gwfv_ad99)

print("Differences calculated. Now calc MAE and RMSE.")

# Calculate MAE and RMSE
## We will take latitude weighted means
weights = np.cos(np.radians(lat))

offline_gwfu_abs_diff = abs(offline_gwfu_diff)
offline_gwfv_abs_diff = abs(offline_gwfv_diff)
online_gwfu_abs_diff = abs(online_gwfu_diff)
online_gwfv_abs_diff = abs(online_gwfv_diff)

offline_gwfu_sq_diff = offline_gwfu_diff**2
offline_gwfv_sq_diff = offline_gwfv_diff**2
online_gwfu_sq_diff = online_gwfu_diff**2
online_gwfv_sq_diff = online_gwfv_diff**2


print("Abs and sq errors calculated. Now saving global and tropical means")
## What to save as
save_filestart = filename.removesuffix(".nc")
save_as = f"{online_dir}/{model_start}_seed{seed}/{save_filestart}_online_offline_errs.nc"

ds = offline_gwfu_diff.to_dataset(name = f"offline_gwfu_diff")
ds[f"offline_gwfv_diff"] = offline_gwfv_diff
ds[f"online_gwfu_diff"] = online_gwfu_diff
ds[f"online_gwfv_diff"] = online_gwfv_diff

## Save metrics: global
ds[f"offline_gwfu_MAE"] = offline_gwfu_abs_diff.weighted(
        weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE"] = offline_gwfv_abs_diff.weighted(
        weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE"] = online_gwfu_abs_diff.weighted(
        weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE"] = online_gwfv_abs_diff.weighted(
        weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE"] = ( offline_gwfu_sq_diff.weighted(
        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE"] = ( offline_gwfv_sq_diff.weighted(
        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE"] =  ( online_gwfu_sq_diff.weighted(
        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE"] =  ( online_gwfv_sq_diff.weighted(
        weights).mean(dim=["lat","lon"])  )**0.5

## Equator: -5 to 5 deg
ds[f"offline_gwfu_MAE_eq"] = offline_gwfu_abs_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_eq"] = offline_gwfv_abs_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_eq"] = online_gwfu_abs_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_eq"] = online_gwfv_abs_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_eq"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_eq"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_eq"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_eq"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(-5,5) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5

## Tropics: -15 to 15 deg
ds[f"offline_gwfu_MAE_tropics"] = offline_gwfu_abs_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_tropics"] = offline_gwfv_abs_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_tropics"] = online_gwfu_abs_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_tropics"] = online_gwfv_abs_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_tropics"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_tropics"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_tropics"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_tropics"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(-15,15) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5


## NHML
ds[f"offline_gwfu_MAE_NHML"] = offline_gwfu_abs_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_NHML"] = offline_gwfv_abs_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_NHML"] = online_gwfu_abs_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_NHML"] = online_gwfv_abs_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_NHML"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_NHML"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_NHML"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_NHML"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(30,60) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5

## NHHL
ds[f"offline_gwfu_MAE_NHHL"] = offline_gwfu_abs_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_NHHL"] = offline_gwfv_abs_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_NHHL"] = online_gwfu_abs_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_NHHL"] = online_gwfv_abs_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_NHHL"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_NHHL"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_NHHL"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_NHHL"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(60,90) ).weighted(
                            weights).mean(dim=["lat","lon"])  )**0.5


## SHML
ds[f"offline_gwfu_MAE_SHML"] = offline_gwfu_abs_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_SHML"] = offline_gwfv_abs_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_SHML"] = online_gwfu_abs_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                            weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_SHML"] = online_gwfv_abs_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                            weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_SHML"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_SHML"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_SHML"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_SHML"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(-60,-30) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5

## SHHL
ds[f"offline_gwfu_MAE_SHHL"] = offline_gwfu_abs_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"offline_gwfv_MAE_SHHL"] = offline_gwfv_abs_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfu_MAE_SHHL"] = online_gwfu_abs_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])
ds[f"online_gwfv_MAE_SHHL"] = online_gwfv_abs_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])

ds[f"offline_gwfu_RMSE_SHHL"] = ( offline_gwfu_sq_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"offline_gwfv_RMSE_SHHL"] = ( offline_gwfv_sq_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"]) )**0.5
ds[f"online_gwfu_RMSE_SHHL"] =  ( online_gwfu_sq_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5
ds[f"online_gwfv_RMSE_SHHL"] =  ( online_gwfv_sq_diff.sel(
                    lat=slice(-90,-60) ).weighted(
                        weights).mean(dim=["lat","lon"])  )**0.5

ds.to_netcdf(save_as)
print(f"`Done. Saved as {save_as}")



