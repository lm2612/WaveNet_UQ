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

# Dataset NH
ds_NH = ds.sel(lat=slice(50,90))
ds_SH = ds.sel(lat=slice(-90,-50))

# Zonal mean across all variables
ds_NHPV = ds_NH.mean(("lon"))
ds_SHPV = ds_SH.mean(("lon"))

# Extend time series to all years
for t in range(t_start+1, t_end):
    try:
        ds = xr.open_dataset(f"{base_dir}/{dir_name}/atmos_daily_{t}.nc", decode_times=False)
        ds_NH = ds.sel(lat=slice(50,90))
        ds_NHPV = xr.concat( (ds_NHPV, ds_NH.mean(("lon")) ), dim="time")

        ds_SH = ds.sel(lat=slice(-90,-50))
        ds_SHPV =  xr.concat( (ds_SHPV, ds_SH.mean(("lon")) ), dim="time")
    except:
        print(f"no file for year {t}, assume model was restart to year {t+1}")

# Save
print(ds_NH)
file_out = f"{base_dir}/{dir_name}/PV_NH_winds{file_num}.nc"
ds_NHPV.to_netcdf(path=file_out, mode='w')
print(f"Saved to {file_out}")

print(ds_SH)
file_out = f"{base_dir}/{dir_name}/PV_SH_winds{file_num}.nc"
ds_SHPV.to_netcdf(path=file_out, mode='w')
print(f"Saved to {file_out}")

