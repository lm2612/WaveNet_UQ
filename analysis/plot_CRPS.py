import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


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

subsample_time = 1 
subsample_lat = 1
subsample_lon = 1

# Subsample
gwfu_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon//subsample_lon ))
gwfv_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon//subsample_lon ))

gwfu_ad99 = gwfu_ad99[::subsample_time, :, ::subsample_lat, ::subsample_lon].to_numpy()
gwfv_ad99 = gwfv_ad99[::subsample_time, :, ::subsample_lat, ::subsample_lon].to_numpy()

lat = lat[::subsample_lat]
lon = lon[::subsample_lon]
time = time[::subsample_time]

for n, seed in enumerate(seeds):
    ## OFFLINE
    ## zonal component
    component = "zonal"
    ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
    ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
    gwfu_preds[n] = ds_u["gwfu_cgwd_pred"][:ntime:subsample_time, :, ::subsample_lat, ::subsample_lon]
    ## meridional component
    component = "meridional"
    ML_dir = f"{offline_dir}/{model_start}_{component}_seed{seed}/"
    ds_u = xr.open_dataset(f"{ML_dir}/{filename}", decode_times=False )
    gwfv_preds[n] = ds_u["gwfv_cgwd_pred"][:ntime:subsample_time, :, ::subsample_lat, ::subsample_lon]

## Reshape: We want data points to be on first axes (lon and time axes merge)
gwfu_preds = np.moveaxis(gwfu_preds, -1, 1).reshape((n_seeds, (ntime//subsample_time)*(nlon//subsample_lon), npfull, nlat//subsample_lat) )
gwfv_preds = np.moveaxis(gwfv_preds, -1, 1).reshape((n_seeds, (ntime//subsample_time)*(nlon//subsample_lon), npfull, nlat//subsample_lat) )
gwfu_ad99 = np.moveaxis(gwfu_ad99, -1, 0).reshape(( (ntime//subsample_time)*(nlon//subsample_lon), npfull, nlat//subsample_lat) )
gwfv_ad99 = np.moveaxis(gwfv_ad99, -1, 0).reshape(( (ntime//subsample_time)*(nlon//subsample_lon), npfull, nlat//subsample_lat) )
   

def crps(y_true, y_pred, sample_weight=None, norm=False):
    """Calculate Continuous Ranked Probability Score
    Data based on size (N, np_full) where N=number of samples (in time, lat, lon) and np_full is height (=40) 
    Args:
     * y_true : np.array (N, np_full) ground truth from AD99 for N samples, for profile of height np_full
     * y_pred : np.array (n_seeds, N, np_full) predicted from n_seeds ensembles, for N samples, for profile of height np_full
    """
    num_samples = y_pred.shape[0]
    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, (-2,-1))
    ##y_true = np.expand_dims(y_true, 0)
    absolute_error = np.mean(np.abs(y_pred - np.expand_dims(y_true, 0)), axis=(0)) 
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    if norm:
        crps_normalized = np.where(np.abs(y_true)> 1E-14, per_obs_crps/np.abs(y_true), np.nan)
        return np.nanmean(crps_normalized, axis=0)
    return np.average(per_obs_crps, axis=0, weights=sample_weight)

# CRPS plot
plt.clf()
# Select latitude bands for subseting data and plotting
labels = ["<85$\degree$S", "55$\degree$-65$\degree$S", "25$\degree$-35$\degree$S",
          "5$\degree$S-5$\degree$N", "25$\degree$-35$\degree$N", "55$\degree$-65$\degree$N", ">85$\degree$N"]
lat_inds = [slice(0, 3), slice(9,12), slice(19,23), slice(30,34),
    slice(41,45), slice(52,55), slice(62, 64)]
cmap = cm.coolwarm(np.linspace(0, len(lat_inds)))

fig, axs = plt.subplots(1, 2 , figsize=(10, 5), sharey = True)
## Zonal
ax = axs[0]
plt.sca(ax)
for i, lat_ind in enumerate(lat_inds):
    # Select all gwfu in this latitude range
    gwfu_ad99_lat = gwfu_ad99[..., lat_ind]
    # Reshape to (N, npfull)
    nlat_ind = gwfu_ad99_lat.shape[-1]
    gwfu_ad99_lat = np.moveaxis(gwfu_ad99[..., lat_ind], -1, 0).reshape((ntime*nlon*nlat_ind, npfull))
    gwfu_preds_lat = np.moveaxis(gwfu_preds[..., lat_ind], -1, 1).reshape((n_seeds, ntime*nlon*nlat_ind, npfull))
    # CRPS
    crps_u = crps(gwfu_ad99_lat, gwfu_preds_lat)
    # Plot
    plt.semilogy(crps_u, pfull,  color=cmap[i], label=labels[i])
# Labels, axis, legends
plt.xlabel("CRPS for Zonal GWD (ms$^{-2}$)")
plt.title("Zonal")
ax.invert_yaxis()
plt.ylabel("Pressure (hPa)")
plt.legend()
plt.axis(xmin=0, xmax=5e-6, ymin=3e2, ymax=1)
# Add a) label
plt.text(x=-0.15, y=1.01, s="a)", fontsize=16, transform=ax.transAxes)

## Meridional
ax = axs[1]
plt.sca(ax)
for i, lat_ind in enumerate(lat_inds):
    # Select all gwfv in this latitude range
    gwfv_ad99_lat = gwfv_ad99[..., lat_ind]
    # Reshape to (N, npfull)
    nlat_ind = gwfv_ad99_lat.shape[-1]
    gwfv_ad99_lat = np.moveaxis(gwfv_ad99[..., lat_ind], -1, 0).reshape((ntime*nlon*nlat_ind, npfull))
    gwfv_preds_lat = np.moveaxis(gwfv_preds[..., lat_ind], -1, 1).reshape((n_seeds, ntime*nlon*nlat_ind, npfull))
    # CRPS
    crps_v = crps(gwfv_ad99_lat, gwfv_preds_lat)
    # Plot
    plt.semilogy(crps_v, pfull,  color=cmap[i], label=labels[i])
# Labels, axis, legends
plt.xlabel("CRPS for Meridional GWD (ms$^{-2}$)")
plt.title("Meridional")
plt.legend()
plt.axis(xmin=0, xmax=5e-6) 
# Add b) label
plt.text(x=-0.1, y=1.01, s="b)", fontsize=16, transform=ax.transAxes)

plt.suptitle("Continuous Ranked Probability Score")
save_plotname = f"{save_dir}/GWD_err_CRPS_profile.png"
plt.savefig(save_plotname)

print(f"Done saved as {save_plotname}")
           

# Normalized crps
plt.clf()
fig, axs = plt.subplots(1, 2 , figsize=(10, 5), sharey = True)

## Zonal
ax = axs[0]
plt.sca(ax)
for i, lat_ind in enumerate(lat_inds):
    # Select all gwfu in this latitude range
    gwfu_ad99_lat = gwfu_ad99[..., lat_ind]
    # Reshape to (N, npfull)
    nlat_ind = gwfu_ad99_lat.shape[-1]
    gwfu_ad99_lat = np.moveaxis(gwfu_ad99[..., lat_ind], -1, 0).reshape((ntime*nlon*nlat_ind, npfull))
    gwfu_preds_lat = np.moveaxis(gwfu_preds[..., lat_ind], -1, 1).reshape((n_seeds, ntime*nlon*nlat_ind, npfull))
    # CRPS
    crps_u = crps(gwfu_ad99_lat, gwfu_preds_lat, norm=True)
    # Plot
    plt.semilogy(crps_u, pfull,  color=cmap[i], label=labels[i])
# Labels, axis, legends
plt.xlabel("Normalized CRPS for Zonal GWD")
plt.title("Zonal")
ax.invert_yaxis()
plt.ylabel("Pressure (hPa)")
plt.legend()
#plt.axis(xmin=0, xmax=5e-6, ymin=3e2, ymax=1)
plt.axis(xmin=0, ymin=3e2, ymax=1, xmax=30)
# Add a) label
plt.text(x=-0.15, y=1.01, s="a)", fontsize=16, transform=ax.transAxes)

## Meridional
ax = axs[1]
plt.sca(ax)
for i, lat_ind in enumerate(lat_inds):
    # Select all gwfv in this latitude range
    gwfv_ad99_lat = gwfv_ad99[..., lat_ind]
    # Reshape to (N, npfull)
    nlat_ind = gwfv_ad99_lat.shape[-1]
    gwfv_ad99_lat = np.moveaxis(gwfv_ad99[..., lat_ind], -1, 0).reshape((ntime*nlon*nlat_ind, npfull))
    gwfv_preds_lat = np.moveaxis(gwfv_preds[..., lat_ind], -1, 1).reshape((n_seeds, ntime*nlon*nlat_ind, npfull))
    # CRPS
    crps_v = crps(gwfv_ad99_lat, gwfv_preds_lat, norm=True)
    # Plot
    plt.semilogy(crps_v, pfull,  color=cmap[i], label=labels[i])
# Labels, axis, legends
plt.xlabel("Normalized CRPS for Meridional GWD")
plt.title("Meridional")
plt.legend()
#plt.axis(xmin=0, xmax=5e-6)
plt.axis(xmin=0, xmax = 30)
# Add b) label
plt.text(x=-0.1, y=1.01, s="b)", fontsize=16, transform=ax.transAxes)

plt.suptitle("Continuous Ranked Probability Score")
save_plotname = f"{save_dir}/GWD_err_normalized_CRPS_profile.png"
plt.savefig(save_plotname)

print(f"Done saved as {save_plotname}")
    

