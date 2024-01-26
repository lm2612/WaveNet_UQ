import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('../../MiMA_analysis')
from clim_functions.seasons import get_seasonal_inds


###### Set up directories and files #####

#### ML online and offline dirs #### 
offline_dir = "/scratch/users/lauraman/WaveNetPyTorch/models/"
online_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/"
model_start = "wavenet_1"
save_dir = f"{online_dir}/PLOTS/"
t_range = range(45, 66) ## TEMP
n_t = len(t_range)
filenames = [f"atmos_daily_{t}.nc" for t in t_range]

ad99_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"

filename = f"atmos_daily_{t_range[0]}.nc"
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

for filename in filenames[1:]:
    ds_ad99 = xr.concat((ds_ad99,
        xr.open_dataset(f"{ad99_dir}/{filename}", decode_times=False )
        ), dim="time")


# Set seeds
model_start = "wavenet_1"
seeds = list(range(100,121))
n_seeds = len(seeds)
seed_inds = np.arange(n_seeds)
ntime = 360*n_t
npfull = 40
print(ntime)
regions = {"Equator": slice(-5, 5),
           "Tropics": slice(-15, 15),
           "NHML":slice(30,60),
           "NHHL":slice(60,90),
           "SHML":slice(-60,-30),
           "SHHL":slice(-90,-60),
           "Global":slice(-90,90) }
region_names = list(regions.keys())
DJF_inds, MAM_inds, JJA_inds, SON_inds = get_seasonal_inds(ntime)
ANN_inds = list(range(ntime))
seasons = {
        "DJF": DJF_inds,
        "MAM": MAM_inds,
        "JJA": JJA_inds,
        "SON":SON_inds, 
        "ANN": ANN_inds}
seasons = {"ANN":ANN_inds}
season_names = list(seasons.keys())


levs = [13, 17, 23] 
npfull_reg = npfull

def plot_hist(true, online_pred, offline_pred=None, ax=None,
        xlabel="", title=""):
    if ax is None:
        fig, ax = plt.subplot(1,1, figsize=(5, 4))
    plt.sca(ax)
    if offline_pred is not None:
        plt.hist(offline_pred.flatten(), bins=30, color="blue",
                histtype="step", density=True, label="Offline")
    plt.hist(online_pred.flatten(), bins=30, color="red",
            histtype="step", density=True, label="Online")
    plt.hist(true.flatten(), bins=30, color="black",
            histtype="step", density=True, label="AD99")

    #ax.set_yscale("log")
    plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)
    return ax

def plot_distribution(true, online_pred, offline_pred=None, ax=None,
        xlabel="", title="", x=np.arange(-3e-5, 3.2e-5, 1e-7)):
    if ax is None:
        fig, ax = plt.subplot(1,1, figsize=(5, 4))
    plt.sca(ax)
    ## Plot truth first
    kde = stats.gaussian_kde(true.flatten())
    ax.plot(x, kde(x), color="black", alpha=0.8 , label="AD99")
    ## Offline
    if offline_pred is not None:
        kde = stats.gaussian_kde(offline_pred.flatten())
        ax.plot(x, kde(x), color="blue", alpha=0.8, label="Offline")
    ## Online
    kde = stats.gaussian_kde(online_pred.flatten())
    ax.plot(x, kde(x), color="red",  alpha=0.8, label="Online")
    plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)
    return ax



for region_name in region_names:
    print(region_name)
    reg_slice = regions[region_name]
    lat_reg = lat.sel(lat=reg_slice)
    nlat_reg = len(lat_reg)
   
    true_gwfu = ds_ad99["gwfu_cgwd"].sel(lat = reg_slice).to_numpy()
    true_gwfv = ds_ad99["gwfv_cgwd"].sel(lat = reg_slice).to_numpy()
    true_u = ds_ad99["ucomp"].sel(lat = reg_slice).to_numpy()
    true_v = ds_ad99["vcomp"].sel(lat = reg_slice).to_numpy()

    offline_gwfu_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    offline_gwfv_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_gwfu_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_gwfv_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_u_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_v_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))

    for n, seed in enumerate(seeds):
        ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
        print(ML_dir)
        for ti, t in enumerate(t_range):
            ## Offline zonal and meridional
            ML_dir = f"{offline_dir}/{model_start}_zonal_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False )
            offline_gwfu_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfu_cgwd_pred"].sel(lat=reg_slice)

            ML_dir = f"{offline_dir}/{model_start }_meridional_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False )
            offline_gwfv_preds[n, ti*360 : (ti+1)*360] = ds_u["gwfv_cgwd_pred"].sel(lat=reg_slice)
            ## ONLINE
            ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False )
            online_gwfu_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfu_cgwd"].sel(lat=reg_slice)
            online_gwfv_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfv_cgwd"].sel(lat=reg_slice)
            online_u_preds[n, ti*360 : (ti+1)*360] = ds_u[f"ucomp"].sel(lat=reg_slice)
            online_v_preds[n, ti*360 : (ti+1)*360] = ds_u[f"vcomp"].sel(lat=reg_slice)

  
    for lev in levs:
        for season_name in season_names:
            print(season_name)
            season_inds = seasons[season_name]
            ### Plot GWFU
            fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
            plot_distribution(true_gwfu[season_inds, lev], 
                    online_gwfu_preds[:, season_inds, lev], 
                    offline_gwfu_preds[:, season_inds, lev],
                    ax=axs[0], xlabel="Zonal GWD (m/s^2)", title="Zonal", x=np.arange(-1e-5, 1.01e-5, 2e-7))
            plot_distribution(true_gwfv[season_inds, lev], 
                    online_gwfv_preds[:, season_inds, lev], 
                    offline_gwfv_preds[:, season_inds, lev],
                    ax=axs[1], xlabel="Meridional GWD (m/s^2)", title="Meridional", x=np.arange(-1e-5, 1.01e-5, 2e-7))

            if season_name == "ANN":
                plt.suptitle(f"Distributions of Gravity Wave Drag for {region_name} at {pfull[lev]:.1f} hPa")

            else:
                plt.suptitle(f"Distributions of Gravity Wave Drag for {season_name} in {region_name} at {pfull[lev]:.1f} hPa")

            plt.tight_layout()
            save_as = f"{save_dir}/Distributions_of_GWD_{season_name}_{region_name}_lev{lev}.png"
            plt.savefig(save_as)
            print(f"Saved as {save_as}")

            plt.clf()

            ### Plot wind
            fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
            plot_distribution(true_u[season_inds, lev], online_u_preds[:, season_inds, lev],
                            ax=axs[0], xlabel="Zonal wind (m/s)", title="Zonal", x=np.arange(-60, 60.1, 5) )
            plot_distribution(true_v[season_inds, lev], online_v_preds[:, season_inds, lev], 
                    ax=axs[1], xlabel="Meridional wind (m/s)", title="Meridional",  x=np.arange(-60, 60.1, 5) )

            if season_name == "ANN":
                plt.suptitle(f"Distributions of Wind for {region_name} at {pfull[lev]:.1f} hPa")
            else:
                plt.suptitle(f"Distributions of Wind for {season_name} in {region_name} at {pfull[lev]:.1f} hPa")
            plt.tight_layout()
            save_as = f"{save_dir}/Distributions_of_wind_{season_name}_{region_name}_lev{lev}.png"
            plt.savefig(save_as)
            print(f"Saved as {save_as}")
            
            ## HISTS
            plt.clf()
            fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
            plot_hist(true_gwfu[season_inds, lev],
                    online_gwfu_preds[:, season_inds, lev],
                    offline_gwfu_preds[:, season_inds, lev],
                    ax=axs[0], xlabel="Zonal GWD (m/s^2)", title="Zonal")
            plot_hist(true_gwfv[season_inds, lev],
                    online_gwfv_preds[:, season_inds, lev],
                    offline_gwfv_preds[:, season_inds, lev],
                    ax=axs[1], xlabel="Meridional GWD (m/s^2)", title="Meridional")

            plt.suptitle(f"Histogram of Gravity Wave Drag for {season_name} in {region_name} at {pfull[lev]:.1f} hPa")
            plt.tight_layout()
            save_as = f"{save_dir}/Histogram_of_GWD_{season_name}_{region_name}_lev{lev}.png"
            plt.savefig(save_as)
            print(f"Saved as {save_as}")

            plt.clf()
            fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
            plot_hist(true_u[season_inds, lev],
                    online_u_preds[:, season_inds, lev],
                    ax=axs[0], xlabel="Zonal wind (m/s)", title="Zonal")
            plot_hist(true_v[season_inds, lev],
                    online_v_preds[:, season_inds, lev],
                    ax=axs[1], xlabel="Meridional wind (m/s)", title="Meridional")

            plt.suptitle(f"Histogram of Wind for {season_name} in {region_name} at {pfull[lev]:.1f} hPa")
            plt.tight_layout()
            save_as = f"{save_dir}/Histogram_of_wind_{season_name}_{region_name}_lev{lev}.png"
            plt.savefig(save_as)
            print(f"Saved as {save_as}")




