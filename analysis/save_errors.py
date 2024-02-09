import numpy as np
import xarray as xr
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--year_start', type=int, default=45)

args = parser.parse_args()
seed = args.seed
year_start = args.year_start

###### Set up directories and files #####

##### AD99 run #####
ad99_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"

filename = f"atmos_daily_{year_start}.nc"

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


#### ML runs: offline predictions ####
online_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/"
offline_dir = "/scratch/users/lauraman/WaveNetPyTorch/models/"
model_start = "wavenet_1"

save_dir = f"{online_dir}/PLOTS/"

# Set seeds
seeds = range(100, 130)
n_seeds = len(seeds)

## OFFLINE
## zonal component
component = "zonal"
ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
gwfu_preds = ds_u["gwfu_cgwd_pred"]
## meridional component
component = "meridional"
ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
gwfv_preds = ds_u["gwfv_cgwd_pred"]

## Error metrics: MAE, RMSE, R2, CRPS
## arrays are of size (N_ensemble, N_samples (time/lat/lon), pfull)
def MAE(y_true, y_pred, sample_weight=None):
    if y_pred.ndim > y_true.ndim:
        y_pred = np.mean(y_pred, axis=0) 
    y_pred = np.mean(y_pred, axis=0)
    mean_absolute_error = np.average(np.abs(y_pred - y_true), axis=0, weights=sample_weight)
    return mean_absolute_error

def RMSE(y_true, y_pred, sample_weight=None):
    if y_pred.ndim > y_true.ndim:
        y_pred = np.mean(y_pred, axis=0) 
    mean_squared_error = np.average((y_pred - y_true)**2, axis=0, weights=sample_weight)
    return np.sqrt(mean_squared_error)


def r2(y_true, y_pred, sample_weight=None):
    if y_pred.ndim > y_true.ndim:   
        y_pred = np.mean(y_pred, axis=0) 
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true-y_mean)**2, axis=0)
    ss_res = np.sum((y_true-y_pred)**2, axis=0)
    return 1 - (ss_res/ss_tot)

mae_u = np.mean(np.abs(gwfu_preds - gwfu_ad99), axis=0)
mae_v = np.mean(np.abs(gwfv_preds - gwfv_ad99), axis=0)

rmse_u = np.mean(np.sqrt((gwfu_preds - gwfu_ad99)**2), axis=0)
rmse_v = np.mean(np.sqrt((gwfv_preds - gwfv_ad99)**2), axis=0)

u_mean = np.mean(gwfu_ad99)
ss_tot = np.sum((gwfu_ad99 - u_mean)**2, axis=0 )
ss_res = np.sum((gwfu_ad99 - gwfu_preds)**2, axis=0 )
r2_u = 1 - (ss_res/ss_tot)


v_mean = np.mean(gwfv_ad99)
ss_tot = np.sum((gwfv_ad99 - v_mean)**2, axis=0 )
ss_res = np.sum((gwfv_ad99 - gwfv_preds)**2, axis=0 )
r2_v = 1 - (ss_res/ss_tot)

# SAVE
save_filestart = filename.removesuffix(".nc")
save_as = f"{online_dir}/{model_start}_seed{seed}/offline_mae_{save_filestart}.nc"
ds = mae_u.to_dataset(name=f"mae_u")
ds[f"mae_v"] = mae_v
ds.to_netcdf(path=save_as, mode='w')
print(f"Saved to {save_as}")

save_as = f"{online_dir}/{model_start}_seed{seed}/offline_rmse_{save_filestart}.nc"
ds = rmse_u.to_dataset(name=f"rmse_u")
ds[f"rmse_v"] = rmse_v
ds.to_netcdf(path=save_as, mode='w')
print(f"Saved to {save_as}")


