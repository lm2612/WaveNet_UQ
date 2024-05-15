import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats

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
seeds = range(100, 130)
n_seeds = len(seeds)

# Subsample for cost reasons
subsample_time = 10 
subsample_lat = 1

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


# Select latitude bands for subseting data and plotting
labels = ["<85$\degree$S", "55$\degree$-65$\degree$S", "25$\degree$-35$\degree$S",
                  "5$\degree$S-5$\degree$N", "25$\degree$-35$\degree$N", "55$\degree$-65$\degree$N", ">85$\degree$N"]
lat_inds = [slice(0, 3), slice(9,12), slice(19,23), slice(30,34),
            slice(41,45), slice(52,55), slice(62, 64)]
# Plot for a given height, lat and lon sample
mlevs = [13, 17, 20, 23]
# We will plot scatter points with more density in a darker color
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "darkorange", "red", "darkred"])
for mlev in mlevs:
    plev = pfull[mlev]
    for j, lat_ind in enumerate(lat_inds):
        plt.clf()
        fig, axs = plt.subplots(1, 2 , figsize=(10, 4))
        # Zonal
        errs = gwfu_errs[:, mlev, lat_ind, :].flatten()
        sigma = gwfu_sigma[:, mlev, lat_ind, :].flatten()
        # Calculate the point density for colors
        xy = np.vstack([errs, sigma])
        z = stats.gaussian_kde(xy)(xy)
        idx = z.argsort()
        errs, sigma, z = errs[idx], sigma[idx], z[idx]

        # Plot
        plt.sca(axs[0])
        axs[0].plot([0., 4.e-6], [0., 4.e-6], color="black", linestyle="dashed")
        axs[0].scatter(errs, sigma, alpha=0.25, c=z, cmap=cmap)
        # Labels, axes, title, etc.
        plt.xlabel("Ensemble Mean Absolute Error (ms$^{-2}$)")
        plt.ylabel("Ensemble 1$\sigma$ Uncertainty (ms$^{-2}$)")
        plt.axis(ymax=3e-6, xmax=6e-6)
        plt.title("Zonal")
        plt.text(x=-0.15, y=1.01, s="a)", fontsize=16, transform=axs[0].transAxes)

        # Meridional
        errs = gwfv_errs[:, mlev, lat_ind, :].flatten()
        sigma = gwfv_sigma[:, mlev, lat_ind, :].flatten()
        # Calculate the point density for colors
        xy = np.vstack([errs, sigma])
        z = stats.gaussian_kde(xy)(xy)
        idx = z.argsort()
        errs, sigma, z = errs[idx], sigma[idx], z[idx]
        # Plot
        plt.sca(axs[1])
        axs[1].plot([0., 4.e-6], [0., 4.e-6], color="black", linestyle="dashed")
        axs[1].scatter(errs, sigma, alpha=0.25, c=z, cmap=cmap)

        # Labels, axes, titles, etc.
        plt.xlabel("Ensemble Mean Absolute Error (ms$^{-2}$)")
        plt.ylabel("Ensemble 1$\sigma$ Uncertainty (ms$^{-2}$)")
        plt.axis(ymax=3e-6, xmax=6e-6)
        plt.title("Meridional")
        plt.text(x=-0.15, y=1.01, s="b)", fontsize=16, transform=axs[1].transAxes)

        plt.suptitle(f"Confidence of neural networks at latitudes {labels[j]} at {plev:.1f} hPa")

        # Save
        save_label = labels[j].replace("$\degree$", "o")
        save_plotname = f"{save_dir}/GWDs_errors_vs_1std_uncertainty_level{mlev}_{save_label}.png"
        plt.tight_layout()
        plt.savefig(save_plotname, bbox_inches="tight")

print(f"Done all plots for lev {mlev}. Saved as {save_plotname}")
           

