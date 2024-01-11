import xarray as xr
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="wavenet_seed1")
args = parser.parse_args()
dir_name = args.dir

base_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs"


print(f"Opening file for {dir_name}")
start_year = 45
end_year = 65


ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{start_year}.nc", decode_times=False)

# Dataset at equator 
ds_tropics = ds.sel(lat=slice(-5,5))
lat = ds_tropics["lat"]
time = ds_tropics["time"]

# Take area-weighted mean across all variables
weights = np.cos(np.deg2rad(lat))
ds_QBO = ds_tropics.weighted(weights).mean(("lon","lat"))
print(ds_QBO)

# Extend time series to all years
for t in range(start_year+1, end_year+1):
    ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{t}.nc", decode_times=False)
    ds_tropics = ds.sel(lat=slice(-5,5))
    ds_QBO = xr.concat( (ds_QBO, ds_tropics.weighted(weights).mean(("lon","lat")) ), dim="time")

# Save
print(ds_QBO)
file_out = f"{base_dir}/{dir_name}/QBO_winds.nc"
ds_QBO.to_netcdf(path=file_out, mode='w')
print(f"Saved to {file_out}")
