import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib


###### Set up directories and files #####

##### AD99 run #####
ad99_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"

filename = "atmos_daily_45.nc"

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
seeds = range(100, 121)
n_seeds = len(seeds)

# Subsample for cost reasons
subsample_time = 10 
subsample_lat = 4

# Subsample
gwfu_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon ))
gwfv_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon ))

gwfu_ad99 = gwfu_ad99[::subsample_time, :, ::subsample_lat, :]
gwfv_ad99 = gwfv_ad99[::subsample_time, :, ::subsample_lat, :]

lat = lat[::subsample_lat]
time = time[::subsample_time]

for n, seed in enumerate(seeds):
    ## OFFLINE
    ## zonal component
    component = "zonal"
    ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
    ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
    gwfu_preds[n] = ds_u["gwfu_cgwd_pred"][:ntime:subsample_time, :, ::subsample_lat, :]
    ## meridional component
    component = "meridional"
    ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
    ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
    gwfv_preds[n] = ds_u["gwfv_cgwd_pred"][:ntime:subsample_time, :, ::subsample_lat, :]
  
# Ensemble 1 std.
gwfu_sigma = np.std(gwfu_preds, axis=0)
gwfv_sigma = np.std(gwfv_preds, axis=0)
# Mean absolute errors
gwfu_errs = np.mean(np.abs(gwfu_preds - gwfu_ad99.to_numpy()), axis=0)
gwfv_errs = np.mean(np.abs(gwfv_preds - gwfv_ad99.to_numpy()), axis=0)

# Plot for a given height, lat and lon sample
mlevs = [13, 17, 20, 23]
for mlev in mlevs:
    plev = pfull[mlev]
    for j in range(0, nlat//subsample_lat):
        plt.clf()
        fig, axs = plt.subplots(1, 2 , figsize=(10, 4))
        plt.sca(axs[0])
        axs[0].plot([0., 4.e-6], [0., 4.e-6], color="black", linestyle="dashed")
        axs[0].scatter(gwfu_errs[:, mlev, j, :], gwfu_sigma[:, mlev, j, :], alpha=0.5, color="orange")
        
        plt.xlabel("Ensemble Error (MAE) in Zonal GWD (m/s^2)")
        plt.ylabel("Ensemble Uncertainty (1$\sigma$) in Zonal GWD (m/s^2)")
        plt.axis(ymax=3e-6)
        plt.title("Zonal")

        plt.sca(axs[1])
        axs[1].plot([0., 4.e-6], [0., 4.e-6], color="black", linestyle="dashed")
        axs[1].scatter(gwfv_errs[:, mlev, j, :], gwfv_sigma[:, mlev, j, :], alpha=0.5, color="orange")
        plt.xlabel("Ensemble Error (MAE) in Meridional GWD (m/s^2)")
        plt.ylabel("Ensemble Uncertainty (1$\sigma$) in Meridional GWD (m/s^2)")
        plt.axis(ymax=3e-6)
        plt.title("Meridional")


        save_plotname = f"{save_dir}/GWDs_errors_vs_1std_uncertainty_level{mlev}_lat{j*subsample_lat}.png"
        plt.savefig(save_plotname)

print(f"Done all plots for lev {mlev}. Saved as {save_plotname}")
           

