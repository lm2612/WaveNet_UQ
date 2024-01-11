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
model_start = "wavenet"
save_dir = f"{online_dir}/PLOTS/"
t_range = range(45, 50) #65)
n_t = len(t_range)
filenames = [f"atmos_daily_{t}.nc" for t in t_range]

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

for filename in filenames[1:]:
    ds_ad99 = xr.concat((ds_ad99,
        xr.open_dataset(f"{ad99_dir}/{filename}", decode_times=False )
        ), dim="time")


# Set seeds
model_start = "wavenet"
seeds = list(range(100,120))
n_seeds = len(seeds)
seed_inds = np.arange(n_seeds)
ntime = 360*n_t
npfull = 40
print(ntime)
regions = {"Equator": slice(-5, 5),
           "Tropics": slice(-15, 15)} 
region_names = list(regions.keys())

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

    ax.set_yscale("log")
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
    print("True data opened. Getting online/offline preds for all seeds")

    for n, seed in enumerate(seeds):
        print(f"Seed {seed}")
        ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
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

    print("All data opened. Plotting...")
  
    for lev in levs:
        ## Use flattened arrays here
        ## Phase W when u > 0
        true_pos_inds = np.flatnonzero(true_u[:, lev] > 0)
        online_pos_inds = np.flatnonzero(online_u_preds[:, :, lev] > 0)

        ## Phase E when u < 0 
        true_neg_inds = np.flatnonzero(true_u[:, lev] < 0)
        online_neg_inds = np.flatnonzero(online_u_preds[:, :, lev] < 0)

        print(true_pos_inds.shape)
        print(online_pos_inds.shape)


        ### Plot GWFU
        fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=True)
        plot_distribution(true_gwfu[:, lev].flatten()[true_pos_inds], 
                online_gwfu_preds[:, :, lev].flatten()[online_pos_inds],
                np.array([offline_gwfu_preds[i, :, lev].flatten()[true_pos_inds] for i in range(n_seeds)]),
                ax=axs[0,0], xlabel="GWD u (m/s^2)", title="Westerly phase: Zonal")

        plot_distribution(true_gwfv[:, lev].flatten()[true_pos_inds],
                online_gwfv_preds[:, :, lev].flatten()[online_pos_inds],
                np.array([offline_gwfv_preds[i, :, lev].flatten()[true_pos_inds] for i in range(n_seeds)]),
                ax=axs[0,1], xlabel="GWD v (m/s^2)", title="Westerly phase: Meridional")

        plot_distribution(true_gwfu[:, lev].flatten()[true_neg_inds],
                online_gwfu_preds[:, :, lev].flatten()[online_neg_inds],
                np.array([offline_gwfu_preds[i, :, lev].flatten()[true_neg_inds] for i in range(n_seeds)]),
                ax=axs[1,0], xlabel="GWD u (m/s^2)", title="Easterly phase: Zonal")

        plot_distribution(true_gwfv[:, lev].flatten()[true_neg_inds],
                online_gwfv_preds[:, :, lev].flatten()[online_neg_inds],
                np.array([offline_gwfv_preds[i, :, lev].flatten()[true_neg_inds] for i in range(n_seeds)]),
                ax=axs[1,1], xlabel="GWD v (m/s^2)", title="Easterly phase: Meridional")

        plt.suptitle(f"Distributions of Gravity Wave Drag in {region_name} at {pfull[lev]:.1f} hPa")
        plt.tight_layout()
        save_as = f"{save_dir}/Distributions_of_GWD_QBOphase_{region_name}_lev{lev}_seeds100-130.png"
        plt.savefig(save_as)
        print(f"Saved as {save_as}")


        plt.clf()
        ### Plot wind
        fig, axs = plt.subplots(2,2, figsize=(10, 8), sharey=True)
        plot_distribution(true_u[:, lev].flatten()[true_pos_inds],
                online_u_preds[:, :, lev].flatten()[online_pos_inds],
                ax=axs[0,0], xlabel="u (m/s)", title="Westerly phase: Zonal", x=np.arange(-80, 80.1, 1))

        plot_distribution(true_v[:, lev].flatten()[true_pos_inds],
                online_v_preds[:, :, lev].flatten()[online_pos_inds],
                ax=axs[0,1], xlabel="v (m/s)", title="Westerly phase: Meridional", x=np.arange(-80, 80.1, 1))

        plot_distribution(true_u[:, lev].flatten()[true_neg_inds],
                online_u_preds[:, :, lev].flatten()[online_neg_inds],
                ax=axs[1,0], xlabel="u (m/s)", title="Easterly phase: Zonal", x=np.arange(-80, 80.1, 1))

        plot_distribution(true_v[:, lev].flatten()[true_neg_inds],
                online_v_preds[:, :, lev].flatten()[online_neg_inds],
                ax=axs[1,1], xlabel="v (m/s^2)", title="Easterly phase: Meridional", x=np.arange(-80, 80.1, 1))

        plt.suptitle(f"Distributions of Wind in {region_name} at {pfull[lev]:.1f} hPa")
        plt.tight_layout()
        save_as = f"{save_dir}/Distributions_of_wind_QBOphase_{region_name}_lev{lev}_seeds100-130.png"
        plt.savefig(save_as)
        print(f"Saved as {save_as}")




