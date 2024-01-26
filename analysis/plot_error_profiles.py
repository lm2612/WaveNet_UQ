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
model_start = "wavenet"

save_dir = f"{online_dir}/PLOTS_wavenet4yr/"

# Set seeds
seeds = range(100, 125)
n_seeds = len(seeds)

subsample_time = 1 
subsample_lat = 16 # 4
subsample_lon = 32 #16

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

def crps(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, (-2,-1))
    y_true = np.expand_dims(y_true, 0)
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=(0)) 
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, axis=0, weights=sample_weight)

def r2(y_true, y_pred, sample_weight=None):
    if y_pred.ndim > y_true.ndim:   
        y_pred = np.mean(y_pred, axis=0) 
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true-y_mean)**2, axis=0)
    ss_res = np.sum((y_true-y_pred)**2, axis=0)
    return 1 - (ss_res/ss_tot)


def plot_error_profile(err_u, err_v, xmin=0, xmax=1):
    fig, axs = plt.subplots(1, 2 , figsize=(10, 5), sharey = True)
    # Zonal
    ax = axs[0]
    plt.sca(ax)
    # Truth
    plt.semilogy(err_u, pfull,  color="black")
    plt.xlabel("Zonal GWD (m/s^2)")
    plt.title("Zonal GWD")
    ax.invert_yaxis()
    plt.ylabel("Pressure (hPa)")
    plt.axis(xmin=xmin, xmax=xmax, ymin=1e2, ymax=1)

    # Meridional
    ax = axs[1]
    plt.sca(ax)
    plt.semilogy(err_v,  pfull,    color="black")
    plt.xlabel("Meridional GWD (m/s^2)")
    plt.title("Meridional GWD")
    plt.axis(xmin=xmin, xmax=xmax)
    return fig


for j in range(0, nlat//subsample_lat):
    for i in range(0, nlon//subsample_lon):
        plt.clf()
        ## Calculate all error metrics and plot
        # MAE
        mae_u = MAE(gwfu_ad99[..., j, i], gwfu_preds[..., j, i])
        mae_v = MAE(gwfv_ad99[..., j, i], gwfv_preds[..., j, i])
        print(mae_u.shape)
        fig = plot_error_profile(mae_u, mae_v, xmin = 0., xmax=1e-5)
        plt.suptitle("Mean Absolute Error")
        save_plotname = f"{save_dir}/GWD_err_MAE_profile_lat{j*subsample_lat}_lon{i*subsample_lon}.png"
        plt.savefig(save_plotname)

        # RMSE
        plt.clf()
        rmse_u = RMSE(gwfu_ad99[..., j, i], gwfu_preds[..., j, i])
        rmse_v = RMSE(gwfv_ad99[..., j, i], gwfv_preds[..., j, i])
        fig = plot_error_profile(rmse_u, rmse_v, xmin = 0., xmax=1e-5)
        plt.suptitle("Root Mean Squared Error")
        save_plotname = f"{save_dir}/GWD_err_RMSE_profile_lat{j*subsample_lat}_lon{i*subsample_lon}.png"
        plt.savefig(save_plotname)
        
        # CRPS
        plt.clf()
        crps_u = crps(gwfu_ad99[..., j, i], gwfu_preds[..., j, i])
        crps_v = crps(gwfv_ad99[..., j, i], gwfv_preds[..., j, i])
        fig = plot_error_profile(crps_u, crps_v, xmin = 0., xmax=1e-5)
        plt.suptitle("Continuous Ranked Probability Score")
        save_plotname = f"{save_dir}/GWD_err_CRPS_profile_lat{j*subsample_lat}_lon{i*subsample_lon}.png"
        plt.savefig(save_plotname)

        # R2
        plt.clf()
        r2_u = r2(gwfu_ad99[..., j, i], gwfu_preds[..., j, i])
        r2_v = r2(gwfv_ad99[..., j, i], gwfv_preds[..., j, i])
        fig = plot_error_profile(r2_u, r2_v, xmin = 0., xmax=1)
        plt.suptitle("R squared")
        save_plotname = f"{save_dir}/GWD_err_R2_profile_lat{j*subsample_lat}_lon{i*subsample_lon}.png"
        plt.savefig(save_plotname)

    print(f"Done all plots for lat {j*subsample_lat}. Last plot was saved as {save_plotname}")
           

