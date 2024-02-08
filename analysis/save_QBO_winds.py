import xarray as xr
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="wavenet_seed100")
parser.add_argument('--year_start', type=int, default=45)     
parser.add_argument('--n_year', type=int, default=21)      # for AD99, process more years
parser.add_argument('--file_num', type=str, default="")    # for AD99, name additional files as 1,2,3,...

args = parser.parse_args()
dir_name = args.dir
file_num = args.file_num 
t_start = args.year_start
n_year = args.n_year
t_end = t_start + n_year

base_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs"


print(f"Opening file for {dir_name}")
ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{t_start}.nc", decode_times=False)

# Dataset at equator 
ds_tropics = ds.sel(lat=slice(-5,5))
lat = ds_tropics["lat"]
time = ds_tropics["time"]

# Take area-weighted mean across all variables
weights = np.cos(np.deg2rad(lat))
ds_QBO = ds_tropics.weighted(weights).mean(("lon","lat"))
print(ds_QBO)

# Extend time series to all years
for t in range(t_start+1, t_end):
    print(t)
    try:
        ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{t}.nc", decode_times=False)
        ds_tropics = ds.sel(lat=slice(-5,5))
        ds_QBO = xr.concat( (ds_QBO, ds_tropics.weighted(weights).mean(("lon","lat")) ), dim="time")
    except:
        print(f"no file for year {t}, assume model was restart to year {t+1}")


# Save
print(ds_QBO)
file_out = f"{base_dir}/{dir_name}/QBO_winds{file_num}.nc"
ds_QBO.to_netcdf(path=file_out, mode='w')
print(f"Saved to {file_out}")

