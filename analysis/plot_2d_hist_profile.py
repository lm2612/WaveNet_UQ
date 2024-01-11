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
ds_ad99 = xr.open_dataset(f"{ad99_dir}/{filename}", decode_times=False ).isel(pfull=slice(12,25))

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
        xr.open_dataset(f"{ad99_dir}/{filename}", decode_times=False ).isel(pfull=slice(12,25))
        ), dim="time")


# Set seeds
model_start = "wavenet"
seeds = list(range(100, 150))
n_seeds = len(seeds)
seed_inds = np.arange(n_seeds)
ntime = 360*n_t
#npfull = 40
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
season_names = list(seasons.keys())


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

def calc_density(gwfu, x=np.arange(-3e-5, 3.2e-5, 1e-7)):
    kde = stats.gaussian_kde(gwfu.flatten())
    return kde(x)


for region_name in region_names:
    print(region_name)
    reg_slice = regions[region_name]
    lat_reg = lat.sel(lat=reg_slice)
    nlat_reg = len(lat_reg)
   
    true_gwfu = ds_ad99["gwfu_cgwd"].sel(lat = reg_slice).to_numpy()
    true_gwfv = ds_ad99["gwfv_cgwd"].sel(lat = reg_slice).to_numpy()
    true_u = ds_ad99["ucomp"].sel(lat = reg_slice).to_numpy()
    true_v = ds_ad99["vcomp"].sel(lat = reg_slice).to_numpy()
    
    x=np.arange(-1e-5, 1.01e-5, 1e-6)
    true_gwfu_dens = np.zeros((npfull_reg, len(x)))
    true_gwfv_dens = np.zeros((npfull_reg, len(x)))
    x_u = np.arange(-50, 50.01, 5)
    true_u_dens = np.zeros((npfull_reg, len(x_u)))
    true_v_dens = np.zeros((npfull_reg, len(x_u)))

    for plev in range(npfull_reg):
        true_gwfu_dens[plev] = calc_density(true_gwfu[:, plev], x)
        true_gwfv_dens[plev] = calc_density(true_gwfv[:, plev], x)
        true_u_dens[plev] = calc_density(true_u[:, plev], x_u)
        true_v_dens[plev] = calc_density(true_v[:, plev], x_u)

    
    offline_gwfu_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    offline_gwfv_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_gwfu_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_gwfv_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_u_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))
    online_v_preds = np.zeros((n_seeds, ntime, npfull_reg, nlat_reg, nlon))

    offline_gwfu_dens = np.zeros((npfull_reg, len(x)))
    offline_gwfv_dens = np.zeros((npfull_reg, len(x))) 
    online_gwfu_dens = np.zeros(( npfull_reg, len(x)))
    online_gwfv_dens = np.zeros((npfull_reg, len(x)))
    online_u_dens = np.zeros((npfull_reg, len(x_u)))
    online_v_dens = np.zeros((npfull_reg, len(x_u)))


    for n, seed in enumerate(seeds):
        ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
        for ti, t in enumerate(t_range):
            ## Offline zonal and meridional
            ML_dir = f"{offline_dir}/{model_start}_zonal_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False ).isel(pfull=slice(12,25))
            offline_gwfu_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfu_cgwd_pred"].sel(lat=reg_slice)

            ML_dir = f"{offline_dir}/{model_start }_meridional_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False ).isel(pfull=slice(12,25))
            offline_gwfv_preds[n, ti*360 : (ti+1)*360] = ds_u["gwfv_cgwd_pred"].sel(lat=reg_slice)
            ## ONLINE
            ML_dir = f"{online_dir}/{model_start}_seed{seed}/"
            ds_u = xr.open_dataset(f"{ML_dir}/atmos_daily_{t}.nc", decode_times=False ).isel(pfull=slice(12,25))
            online_gwfu_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfu_cgwd"].sel(lat=reg_slice)
            online_gwfv_preds[n, ti*360 : (ti+1)*360] = ds_u[f"gwfv_cgwd"].sel(lat=reg_slice)
            online_u_preds[n, ti*360 : (ti+1)*360] = ds_u[f"ucomp"].sel(lat=reg_slice)
            online_v_preds[n, ti*360 : (ti+1)*360] = ds_u[f"vcomp"].sel(lat=reg_slice)

    print("Calculating density")

    for plev in range(npfull_reg):
        offline_gwfu_dens[plev] = calc_density(offline_gwfu_preds[:, :, plev], x)
        offline_gwfv_dens[plev] = calc_density(offline_gwfv_preds[:, :, plev], x)
        online_gwfu_dens[plev] = calc_density(online_gwfu_preds[:, :, plev], x)
        online_gwfv_dens[plev] = calc_density(online_gwfv_preds[:, :, plev], x)
        online_u_dens[plev] = calc_density(online_u_preds[:, :, plev], x_u)
        online_v_dens[plev] = calc_density(online_v_preds[:, :, plev], x_u)

    ### Plot
    ## Truth 
    fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    plt.sca(axs[0])
    plt.contourf(x, pfull, true_gwfu_dens)
    plt.xlabel("Zonal GWD (m/s^2)")
    plt.title("Zonal GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)
    
    plt.sca(axs[1]) 
    plt.contourf(x, pfull, true_gwfv_dens)
    plt.xlabel("Meridional GWD (m/s^2)")
    plt.title("Meridional GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)
    
    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    plt.ylabel("Pressure (hPa)")

    plt.suptitle(f"True gravity wave drag density for {region_name}")
    plt.tight_layout()
    save_as = f"{save_dir}/GWD_2dhist_true_{region_name}_seeds100-150.png"
    plt.savefig(save_as)


    ## Offline
    plt.clf()
    fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    plt.sca(axs[0])
    plt.contourf(x, pfull, offline_gwfu_dens)
    plt.xlabel("Zonal GWD (m/s^2)")
    plt.title("Zonal GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)

    plt.sca(axs[1])
    plt.contourf(x, pfull, offline_gwfv_dens)
    plt.xlabel("Meridional GWD (m/s^2)")
    plt.title("Meridional GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)
    
    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    plt.ylabel("Pressure (hPa)")

    plt.suptitle(f"Offline gravity wave drag density for {region_name}")
    plt.tight_layout()
    save_as = f"{save_dir}/GWD_2dhist_offline_{region_name}_seeds100-150.png"
    plt.savefig(save_as)


    ## Online
    plt.clf()
    fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    plt.sca(axs[0])
    plt.contourf(x, pfull, online_gwfu_dens)
    plt.xlabel("Zonal GWD (m/s^2)")
    plt.title("Zonal GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)

    plt.sca(axs[1])
    plt.contourf(x, pfull, online_gwfv_dens)
    plt.xlabel("Meridional GWD (m/s^2)")
    plt.title("Meridional GWD")
    plt.axis(xmin=-2e-5, xmax=2e-5)
   
    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    plt.ylabel("Pressure (hPa)")

    plt.suptitle(f"Online gravity wave drag density for {region_name}")
    plt.tight_layout()
    save_as = f"{save_dir}/GWD_2dhist_online_{region_name}_seeds100-150.png"
    plt.savefig(save_as)


    ## Truth winds
    fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    plt.sca(axs[0])
    plt.contourf(x_u, pfull, true_u_dens)
    plt.xlabel("u (m/s)")
    plt.title("Zonal wind")
    plt.axis(xmin=-60, xmax=60)

    plt.sca(axs[1])
    plt.contourf(x_u, pfull, true_v_dens)
    plt.xlabel("v (m/s)")
    plt.title("Meridional wind")
    plt.axis(xmin=-60, xmax=60)

    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    plt.ylabel("Pressure (hPa)")

    plt.suptitle(f"True wind density for {region_name}")
    plt.tight_layout()
    save_as = f"{save_dir}/wind_2dhist_true_{region_name}_seeds100-150.png"
    plt.savefig(save_as)


     ## Online
    plt.clf()
    fig, axs = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    plt.sca(axs[0])
    plt.contourf(x_u, pfull, online_u_dens)
    plt.xlabel("u (m/s)")
    plt.title("Zonal wind")
    plt.axis(xmin=-2e-5, xmax=2e-5)

    plt.sca(axs[1])
    plt.contourf(x_u, pfull, online_v_dens)
    plt.xlabel("v (m/s)")
    plt.title("Meridional wind")
    plt.axis(xmin=-2e-5, xmax=2e-5)

    axs[0].set_yscale('log')
    axs[0].invert_yaxis()
    plt.ylabel("Pressure (hPa)")

    plt.suptitle(f"Online wind density for {region_name}")
    plt.tight_layout()
    save_as = f"{save_dir}/wind_2dhist_online_{region_name}_seeds100-150.png"
    plt.savefig(save_as)
