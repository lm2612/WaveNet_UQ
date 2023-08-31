import xarray as xr
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--seed', metavar='seed', type=int, nargs='+', default=1)
args = parser.parse_args()

base_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs"

for seed in args.seed:
    dir_name = f"wavenet_4yr_seed{seed}"
    print(f"Seed: {seed}. Dir{dir_name}")

    print(f"Opening file for {dir_name}")
    n_year = len(glob.glob(f"{base_dir}/{dir_name}/atmos_daily_*.nc"))
    print(f"{n_year} files")            
    ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_0.nc", decode_times=False)

    # Dataset at equator 
    ds_tropics = ds.sel(lat=slice(-5,5))
    lat = ds_tropics["lat"]
    time = ds_tropics["time"]

    # Take area-weighted mean across all variables
    weights = np.cos(np.deg2rad(lat))
    ds_QBO = ds_tropics.weighted(weights).mean(("lon","lat"))
    print(ds_QBO)

    # Extend time series to all years
    for t in range(1,n_year):
        ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{t}.nc", decode_times=False)
        ds_tropics = ds.sel(lat=slice(-5,5))
        ds_QBO = xr.concat( (ds_QBO, ds_tropics.weighted(weights).mean(("lon","lat")) ), dim="time")

    # Save
    print(ds_QBO)
    file_out = f"{base_dir}/{dir_name}/QBO_winds.nc"
    ds_QBO.to_netcdf(path=file_out, mode='w')
    print(f"Saved to {file_out}")
