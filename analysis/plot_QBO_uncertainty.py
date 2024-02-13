import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append('../analysis/')
from get_QBO_TT_metrics import *

print("UQ plots QBO")
def all_periods_amplitudes(dirname, plev=10, filename="QBO_winds.nc"):
    if type(filename)==list:
        filename0 = filename[0]
        ds = xr.open_dataset(f"{dirname}{filename0}")
        u = ds["ucomp"]
        n_files = len(filename)
        if n_files > 1:
            for t in range(1,len(filename)):
                ds = xr.open_dataset(f"{dirname}{filename[t]}")
                u = xr.concat((u, ds["ucomp"]), dim="time")

    else:
        ds = xr.open_dataset(f"{dirname}{filename}")
        u = ds["ucomp"]
    periods, amplitudes = get_QBO_periods_amplitudes(u.sel(pfull=plev, method="nearest"), N_smooth=30*5)
    return periods, amplitudes


plev = 10
## Set seeds
seeds = list(range(100,130)) 
n_seeds = len(seeds)

## Directories
# AD99: we have 80 years per file, named QBO_winds1.nc, QBO_winds2.nc, etc.
ad99_dir="/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"
ad99_filenames = [f"QBO_winds{n}.nc" for n in range(1,5)]
# ML: only 20 years per file, named QBO_winds.nc but we have one per seed
model_start="wavenet_1"
ML_dirs = [f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/{model_start}_seed{seed}/" for seed in seeds]
save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/PLOTS/"

# Set up dictionaries to save all QBO periods and amplitudes
all_periods = {}
all_amps = {}

# Get QBO periods and amplitudes for AD99 run
periods, amplitudes = all_periods_amplitudes(ad99_dir, plev, filename=ad99_filenames )
AD99_periods = np.array(periods)
AD99_amps = np.array(amplitudes)
all_periods["ad99"] = periods
all_amps["ad99"] = amplitudes

# Get QBO periods and amplitudes for all ML runs
ML_periods = []
ML_amps = []

for i, dirname in enumerate(ML_dirs):
    periods, amplitudes = all_periods_amplitudes(dirname, plev)
    j = 0 
    while j < len(periods):
        if (periods[j] < 10) or (periods[j] > 45):
            periods = np.delete(periods, j)
            amplitudes = np.delete(amplitudes, j)
        else:
            j = j+1 
            

    all_periods[f"seed{i+1}"] = periods
    all_amps[f"seed{i+1}"] = amplitudes

    [ML_periods.append(period) for period in periods]
    [ML_amps.append(amp) for amp in amplitudes]

print("All periods and amplitudes collected. Plotting...")

# Boxplots alone are not that informative: we cannot see the full distribution
# So we plot violin plots with boxplots on top
# Need to define a custom violin plot function for formatting

def custom_violin_plot(ax, data, positions, col1="gray", col2="black"):
    """ Plot horizontal violin plot on axes ax, for data, in position along y-axis
    Args:
        * ax: matplotlib axis
        * data: data points e.g., in list or nd numpy array
        * positions: position to plot 1d for list or nd for numpy array
        * col1: face color of the violin plot (note it will be 50% transparancy)
        * col2: edge color of violin plot
    Returns:
        * dictionary containing all parts of the violin plot (body, edges, etc.)
    """
    parts = ax.violinplot(data,
                          vert=False,
                          positions = positions,
                          widths = 0.8,
                          showmeans=False,
                          showmedians=False,
        showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col1)
        pc.set_edgecolor(col2)
        pc.set_alpha(0.5)


    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    ax.scatter(positions, medians, marker='_', color='white', s=30, zorder=3, alpha=1)

    return parts

## Create horizontal violin plots with box plot overlaid. AD99 at y=1 and ML ensemble at  y=0
plt.clf()
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
# QBO period
plt.sca(axs[0])
# Create violin plots (vp)
vp_ad99 = custom_violin_plot(axs[0],  AD99_periods, positions = [1], col1="gray", col2="black")
vp_ml = custom_violin_plot(axs[0], ML_periods, positions=[0], col1="orange", col2="red")
# Add box plot (bp) on top to show median, interquartile ranges
bp_ad99 = plt.boxplot(AD99_periods, vert=False, patch_artist=True, positions=[1], widths=0.3, zorder=10)
bp_ad99['boxes'][0].set(color="grey", edgecolor="black", alpha=0.5)
bp_ad99['medians'][0].set(color="black", linewidth=3)
bp_ML = plt.boxplot(ML_periods, vert=False, patch_artist=True, positions=[0], widths=0.3, zorder=10)
bp_ML['boxes'][0].set(color="orange", edgecolor="red", alpha=0.5)
bp_ML['medians'][0].set(color="red", linewidth=3)
# Add individual points as scatter points
axs[0].scatter(all_periods["ad99"], np.ones(len(all_periods["ad99"])), color="black", alpha=0.3, zorder=20)
for seed in range(n_seeds):
    p_seed = all_periods[f"seed{seed+1}"]
    axs[0].scatter(p_seed, np.zeros(len(p_seed)), color="red", alpha=0.3, zorder=20)
# Labels, axes, etc
plt.yticks([1, 0], ["AD99","NN"], fontsize=20)
plt.axis(ymin=-1, ymax=2, xmin=20)
plt.xticks(fontsize=16)
plt.xlabel("QBO Period (months)", fontsize=20)
plt.text(x=-0.08 , y=1.0, s="a)", fontsize=20, transform=axs[0].transAxes)

# QBO amplitudes
plt.sca(axs[1])
# Create violin plots (vp)
vp_ad99 = custom_violin_plot(axs[1],  AD99_amps, positions = [1], col1="gray", col2="black")
vp_ml = custom_violin_plot(axs[1], ML_amps, positions=[0], col1="orange", col2="red")
# Add box plot (bp) on top to show median, interquartile ranges
bp_ad99 = plt.boxplot(AD99_amps, vert=False, patch_artist=True, positions=[1], widths=0.3, zorder=10)
bp_ad99['boxes'][0].set(color="grey", edgecolor="black", alpha=0.5)
bp_ad99['medians'][0].set(color="black", linewidth=3)
bp_ML = plt.boxplot(ML_amps, vert=False, patch_artist=True, positions=[0], widths=0.3, zorder=10)
bp_ML['boxes'][0].set(color="orange", edgecolor="red", alpha=0.5)
bp_ML['medians'][0].set(color="red", linewidth=3)
# Add individual points as scatter points
axs[1].scatter(all_amps["ad99"], np.ones(len(all_amps["ad99"])), color="black", alpha=0.3, zorder=20)
for seed in range(n_seeds):
    p_seed = all_amps[f"seed{seed+1}"]
    axs[1].scatter(p_seed, np.zeros(len(p_seed)), color="red", alpha=0.3, zorder=20)
# Labels, axes, etc
plt.yticks([1, 0], ["AD99","NN"], fontsize=20)
plt.xticks(fontsize=16)
plt.xlabel("QBO Amplitude (m/s)", fontsize=20)
plt.axis(ymin=-1, ymax=2, xmin=15)
plt.text(x=-0.08, y=1.0, s="b)", fontsize=20, transform=axs[1].transAxes)

plt.tight_layout()
save_as = f"{save_dir}/QBO_violin_and_boxplots_plev{plev}hPa.png"
plt.savefig(save_as)


print("Plotting done.")
print(f"Plot saved as {save_as}")

N_AD99 = len(AD99_periods)
N_ML = len(ML_periods)
print(f"No of AD99 QBOs : {N_AD99}")
print(f"No of ML QBOs   : {N_ML}")

ML_periods = np.array(ML_periods)
ML_amps = np.array(ML_amps)
mean_AD99_period = np.mean(AD99_periods)
var_AD99_period = np.var(AD99_periods)
std_AD99_period = np.sqrt(var_AD99_period)

mean_ML_period = np.mean(ML_periods)
var_ML_period = np.var(ML_periods)
std_ML_period = np.sqrt(var_ML_period)

mean_AD99_amps = np.mean(AD99_amps)
var_AD99_amps = np.var(AD99_amps)
std_AD99_amps = np.sqrt(var_AD99_amps)

mean_ML_amps = np.mean(ML_amps)
var_ML_amps = np.var(ML_amps)
std_ML_amps = np.sqrt(var_ML_amps)

var_diff_period = np.abs(var_ML_period - var_AD99_period)
std_diff_period = np.sqrt(var_diff_period)
var_diff_amps = np.abs(var_ML_amps - var_AD99_amps)
std_diff_amps = np.sqrt(var_diff_amps)


print(f"          MEAN        VARIABILITY AS 1 STD")
print(f"PERIOD")
print(f"AD99       {mean_AD99_period}       {std_AD99_period}")
print(f"ML       {mean_ML_period}       {std_ML_period}")
print(f"UQ   {std_diff_period}")

print(f"AMP")
print(f"AD99       {mean_AD99_amps}       {std_AD99_amps}")
print(f"ML       {mean_ML_amps}       {std_ML_amps}")
print(f"UQ      {std_diff_amps}")  

print("DONE")
