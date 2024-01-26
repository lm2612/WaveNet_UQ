import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append('../../MiMA_analysis/')

from clim_functions.datetime360 import *
from SSW_metrics.get_SSWs import get_SSWs

print("UQ plots SSW and final warmings")
def pv_stats(rundir, hemisphere="NH", filename="PV_NH_winds.nc"):
    if hemisphere == "NH":
        lat = 60
        NH = True
    else:
        lat = -60
        NH = False

    if type(filename)==list:
        filename0 = filename[0]
        ds = xr.open_dataset(f"{rundir}{filename0}", decode_times=False )
        u10at60 = ds["ucomp"].sel(pfull=10, lat=lat, method="nearest")
        n_files = len(filename)
        if n_files > 1:
            for t in range(1,len(filename)):
                ds = xr.open_dataset(f"{rundir}{filename[t]}", decode_times=False )
                u10at60 = xr.concat((u10at60,
                      ds["ucomp"].sel(pfull=10, lat=lat, method="nearest")
                      ), dim="time")
    else:
         ds = xr.open_dataset(f"{rundir}{filename}", decode_times=False)
         u10at60 = ds["ucomp"].sel(pfull=10, lat=lat, method="nearest")

    time = np.arange(len(u10at60))
    datelist = datetime360(time[:])
    print(len(time))

    nyears = len(datelist)//360
    ssw_dates, spv_dates, pv_init_dates, final_warming_dates = get_SSWs(
                        u10at60, datelist, NH=NH)

    if NH:
        n_ssws = len(ssw_dates) / len(time)
        n_decades = int( len(time) / 3600 )
        n_ssws_per_decade = np.zeros(n_decades)
        ssw_years = (ssw_dates - time[0])/360.
        start_year = 0
        for decade in range(1,n_decades+1):
            end_year = decade * 10
            n_ssws_per_decade[decade-1] = sum(ssw_years[ssw_years > start_year] < end_year)
            start_year = end_year
        return n_ssws_per_decade
    else:
        lifetime = final_warming_dates - pv_init_dates
        return lifetime


plev = 10
## Set seeds
seeds = list(range(100,121))
n_seeds = len(seeds)

## Directories
# AD99: we have 80 years per file, named PV_winds1.nc, PV_winds2.nc, etc.
ad99_dir="/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"
# ML: only 20 years per file, named PV_winds.nc but we have one per seed
model_start="wavenet_1"
ML_dirs = [f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/{model_start}_seed{seed}/" for seed in seeds]
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/PLOTS/"

# Set up dictionaries to save n_ssws_per_decade for NH and PV_lifetime for SH
all_n_ssws_pd_nh = {}
all_pv_lifetime_sh = {}

# NH:
n_ssws_per_decade = pv_stats(ad99_dir, hemisphere="NH", filename=[f"PV_NH_winds{n}.nc" for n in range(1,3)])
AD99_n_ssws_pd = np.array(n_ssws_per_decade)
print(AD99_n_ssws_pd)

all_n_ssws_pd_nh["ad99"] = AD99_n_ssws_pd

# SH:
AD99_lifetime = pv_stats(ad99_dir, hemisphere="SH", filename=[f"PV_SH_winds{n}.nc" for n in range(1,3)])
all_pv_lifetime_sh["ad99"] = AD99_lifetime

# Get PV properties for ML 
ML_n_ssws_pd = []
ML_pv_lifetime = []

for i, rundir in enumerate(ML_dirs):
    print(rundir)
    print("NH")
    # NH
    n_ssws_per_decade = np.array(pv_stats(rundir, hemisphere="NH", filename="PV_NH_winds.nc"))
    print(n_ssws_per_decade)

    print("SH")
    # SH:
    lifetime = pv_stats(rundir, hemisphere="SH", filename="PV_SH_winds.nc")

    all_n_ssws_pd_nh[f"seed{i+1}"] = n_ssws_per_decade
    all_pv_lifetime_sh[f"seed{i+1}"] = lifetime

    [ML_n_ssws_pd.append(nssw) for nssw in n_ssws_per_decade]
    [ML_pv_lifetime.append(lt) for lt in lifetime]

print("All PV properties collected. Plotting...")

plt.clf()
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
plt.sca(axs[0])
bp_ad99 = plt.boxplot(AD99_n_ssws_pd, vert=False, patch_artist=True, positions=[1], widths=0.8)
bp_ad99['boxes'][0].set(color="grey", alpha=0.5)
bp_ad99['medians'][0].set(color="black", linewidth=3)
bp_ML = plt.boxplot(ML_n_ssws_pd, vert=False, patch_artist=True, positions=[0], widths=0.8)
bp_ML['boxes'][0].set(color="orange", alpha=0.5)
bp_ML['medians'][0].set(color="red", linewidth=3)
plt.scatter(all_n_ssws_pd_nh["ad99"], np.ones(len(all_n_ssws_pd_nh["ad99"])), color="gray", alpha=0.5)
for seed in range(n_seeds):
    p_seed = all_n_ssws_pd_nh[f"seed{seed+1}"]
    plt.scatter(p_seed, np.zeros(len(p_seed)), color="red", alpha=0.5)
#plt.axis(xmin=17, xmax=37, ymin=-1, ymax=2)
plt.yticks([1, 0], ["AD99","NN"], fontsize=24)
plt.xticks(fontsize=16)
plt.xlabel("Number of SSWs in NH per decade", fontsize=20)

plt.sca(axs[1])
bp_ad99 = plt.boxplot(AD99_lifetime, vert=False, patch_artist=True, positions=[1], widths=0.8)
bp_ad99['boxes'][0].set(color="grey", alpha=0.5)
bp_ad99['medians'][0].set(color="black", linewidth=3)
bp_ML = plt.boxplot(ML_pv_lifetime, vert=False, patch_artist=True, positions=[0], widths=0.8)
bp_ML['boxes'][0].set(color="orange", alpha=0.5)
bp_ML['medians'][0].set(color="red", linewidth=3)
plt.scatter(all_pv_lifetime_sh["ad99"], np.ones(len(all_pv_lifetime_sh["ad99"])), color="gray", alpha=0.5)
for seed in range(n_seeds):
    p_seed = all_pv_lifetime_sh[f"seed{seed+1}"]
    plt.scatter(p_seed, np.zeros(len(p_seed)), color="red", alpha=0.5)
#plt.axis(xmin=20, xmax=37, ymin=-1, ymax=2)
plt.yticks([1, 0], ["AD99","NN"], fontsize=24)
plt.xticks(fontsize=16)
plt.xlabel("SH polar vortex lifetime (days)", fontsize=20)
plt.tight_layout()
save_as = f"{save_dir}/PV_boxplots_plev{plev}hPa.png"
plt.savefig(save_as)


print("Plotting done.")
print(f"Plot saved as {save_as}")

plt.clf()
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
plt.sca(axs[0])
bp_ad99 = plt.hist(AD99_n_ssws_pd, bins=np.arange(0, 12., 2.), density=True, color="grey", alpha=0.5, label="AD99")
bp_ML = plt.hist(ML_n_ssws_pd, bins=np.arange(0, 12., 2.), density=True, color="orange", alpha=0.5, label="NN")
plt.legend()
plt.xlabel("Number of SSWs per decade (NH)", fontsize=20)

plt.sca(axs[1])
bp_ad99 = plt.hist(AD99_lifetime, density=True, color="grey", alpha=0.5, label="AD99")
bp_ML = plt.hist(ML_pv_lifetime, density=True, color="orange", alpha=0.5, label="NN")
plt.legend()
plt.xlabel("SH polar vortex lifetime (days)", fontsize=20)

plt.tight_layout()
save_as = f"{save_dir}/PV_hist_plev{plev}hPa.png"
plt.savefig(save_as)



print("Plotting done.")
print(f"Plot saved as {save_as}")
