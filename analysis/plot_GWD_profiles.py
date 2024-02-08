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
seeds = range(100, 130)
n_seeds = len(seeds)

subsample_time = 10 
subsample_lat = 4
subsample_lon = 16

# Subsample
gwfu_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon//subsample_lon ))
gwfv_preds = np.zeros((n_seeds, ntime//subsample_time, npfull, nlat//subsample_lat, nlon//subsample_lon ))

gwfu_ad99 = gwfu_ad99[::subsample_time, :, ::subsample_lat, ::subsample_lon]
gwfv_ad99 = gwfv_ad99[::subsample_time, :, ::subsample_lat, ::subsample_lon]

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
   

def plot_GWD_profile(t, j, i):
    fig, axs = plt.subplots(1, 2 , figsize=(10, 5), sharey = True)
    # Zonal
    ax = axs[0]
    plt.sca(ax)
    # Truth
    plt.semilogy(gwfu_ad99[t, :, j, i], 
                 pfull, 
                 color="black", label="True")
    
    colors = matplotlib.cm.jet(np.linspace(0,1,n_seeds))
    for n in range(n_seeds):
        plt.semilogy(gwfu_preds[n, t, :, j, i], 
                pfull,
                color=colors[n],
                alpha=0.7,
                label=f"{n}")
    #plt.legend(loc="lower left")
    plt.xlabel("Zonal GWD (ms$^{-2}$)")
    plt.title("Zonal GWD")
    ax.invert_yaxis()
    plt.ylabel("Pressure (hPa)")
    plt.axis(xmin=-4e-5, xmax=4e-5, ymin=1e2, ymax=1)

    # Meridional
    ax = axs[1]
    plt.sca(ax)
    plt.semilogy(gwfv_ad99[t, :, j, i], 
                 pfull, 
                 color="black", label="True")
    for n in range(n_seeds):
        plt.semilogy(gwfv_preds[n, t, :, j, i],
                pfull,
                color=colors[n],
                alpha=0.7,
                label=f"{n}")
    plt.xlabel("Meridional GWD (m$s^{-2}$)")
    plt.title("Meridional GWD")


    #plt.legend(loc="lower left", ncol=3)
    plt.axis(xmin=-4e-5, xmax=4e-5)

    plt.suptitle(f"{lat[j]:.1f}, {lon[i]:.1f}")
    return fig

def plot_GWD_profile_std(t, j, i):
    fig, axs = plt.subplots(1, 2 , figsize=(10, 5), sharey = True)
    # Zonal
    ax = axs[0]
    plt.sca(ax)
    # Truth
    plt.semilogy(gwfu_ad99[t, :, j, i],
                 pfull,
                 color="black", label="True")
    gwfu_pred_mean = gwfu_preds[:, t, :, j, i].mean(axis=0)
    gwfu_pred_sd = gwfu_preds[:, t, :, j, i].std(axis=0)
    plt.semilogy(gwfu_pred_mean,
                 pfull,
                 color="red",
                 alpha = 1,
                 label="Predicted Mean")
    plt.fill_betweenx(pfull, gwfu_pred_mean - gwfu_pred_sd,
                      gwfu_pred_mean + gwfu_pred_sd ,
                      color="orange", alpha=0.6,
                      label="Predicted 1 $\sigma$")
    plt.legend(loc="lower left")
    plt.xlabel("Zonal GWD (m$s^{-2}$)")
    plt.title("Zonal GWD")
    ax.invert_yaxis()
    plt.ylabel("Pressure (hPa)")
    plt.axis(xmin=-4e-5, xmax=4e-5, ymin=3e2, ymax=1)

    # Meridional
    ax = axs[1]
    plt.sca(ax)
    plt.semilogy(gwfv_ad99[t, :, j, i],
                 pfull,
                 color="black", label="True")
    gwfv_pred_mean = gwfv_preds[:, t, :, j, i].mean(axis=0)
    gwfv_pred_sd = gwfv_preds[:, t, :, j, i].std(axis=0)
    plt.semilogy(gwfv_pred_mean,
                 pfull,
                 color="red",
                 alpha = 1,
                 label="Predicted Mean")
    plt.fill_betweenx(pfull, gwfv_pred_mean - gwfv_pred_sd,
                      gwfv_pred_mean + gwfv_pred_sd ,
                      color="orange", alpha=0.6,
                      label="Predicted 1 $\sigma$")
    plt.xlabel("Meridional GWD (m$s^{-2}$)")
    plt.title("Meridional GWD")


    #plt.legend(loc="lower left", ncol=3)
    plt.axis(xmin=-4e-5, xmax=4e-5)

    plt.suptitle(f"Grid cell: ({lat[j]:.1f}\N{DEGREE SIGN}, {lon[i]:.1f}\N{DEGREE SIGN})")
    return fig


for t in range(0, ntime//subsample_time):
    for j in range(0, nlat//subsample_lat):
        for i in range(0, nlon//subsample_lon):
            plt.clf()
            #fig=plot_GWD_profile(t, j, i)
            #save_plotname = f"{save_dir}/GWDs_profile_ensemble_pred_offline_lat{j*subsample_lat}_lon{i*subsample_lon}_time{t*subsample_time}_seeds100-130.png"
            #plt.savefig(save_plotname)
            plt.clf()
            fig=plot_GWD_profile_std(t, j, i)
            save_plotname = f"{save_dir}/GWDs_profile_mean_std_ensemble_pred_offline_lat{j*subsample_lat}_lon{i*subsample_lon}_time{t*subsample_time:03d}.png"
            plt.savefig(save_plotname)
    print(f"Done all plots for timestep {t*subsample_time}. Saved as {save_plotname}")
           

