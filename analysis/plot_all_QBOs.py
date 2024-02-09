import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append('../analysis/')
from get_QBO_TT_metrics import *
from plot_QBO import plot_QBO

print("Plotting all QBOs")
all_plot_seeds = ["0-9", "10-19", "20-29"]
all_seeds = [list(range(100,110)), list(range(110,120)) , list(range(120,130)) ]

for plot_seeds, seeds in zip(all_plot_seeds, all_seeds):
    print(f"Plot seeds {plot_seeds}")
    plt.clf()
    init_seed = seeds[0]
    n_seeds = len(seeds)

    nyear = 20  # Plot full 20 years
    plev = 10
    ad99_dir="/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"
    ML_dirs = [f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/wavenet_1_seed{seed}/" for seed in seeds]
    save_dir = f"/scratch/users/lauraman/WaveNetPyTorch/mima_runs/PLOTS/"

    ## Plot QBO
    fig, axs = plt.subplots(nrows=n_seeds+1, ncols=1, figsize=(20, 20), sharex=True)
    plt.sca(axs[0])

    ds = xr.open_dataset(f"{ad99_dir}/QBO_winds.nc")
    ds = ds.isel(time=slice(0, nyear*360))
    u = ds["ucomp"]
    pfull = ds["pfull"]
    time = ds["time"]
    periods, amplitudes = get_QBO_periods_amplitudes(u.sel(pfull=plev, method="nearest"), N_smooth=30*5)
    period_mean, amp_mean = np.mean(periods), np.mean(amplitudes)
    period_std, amp_std = np.std(periods), np.std(amplitudes)
    plot_QBO((time-time[0])/360., pfull, u, ax=axs[0], cbar=False, x_axis=False, y_axis=True)
    plt.title(f"AD99. Period={period_mean:.1f} ({period_std:.1f}), Amplitude={amp_mean:.1f} ({amp_std:.1f}) ")
    print(f"AD99 plotted")
    for i, dirname in enumerate(ML_dirs):
        ds = xr.open_dataset(f"{dirname}/QBO_winds.nc")
        ds = ds.isel(time=slice(0, nyear*360))
        u = ds["ucomp"]
        periods, amplitudes = get_QBO_periods_amplitudes(u.sel(pfull=plev, method="nearest"), N_smooth=30*5)
        period_mean, amp_mean = np.mean(periods), np.mean(amplitudes)
        period_std, amp_std = np.std(periods), np.std(amplitudes)
        ## Also print the offline error at the equator to see whether "better" QBOs had lower offline error (turns out they don't)
        ds = xr.open_dataset(f"{dirname}/offline_mae_atmos_daily_45.nc")
        mae_u_eq = ds["mae_u"].sel(lat=slice(-5,5), pfull=slice(10,100))
        mean_err = mae_u_eq.mean()
        plot_QBO((time-time[0])/360., pfull, u, ax=axs[i+1], cbar=False, x_axis=False, y_axis=True)
        plt.title(f"Ensemble Member {i+init_seed}. Period={period_mean:.1f} ({period_std:.1f}), Amplitude={amp_mean:.1f} ({amp_std:.1f}). Offline err={mean_err:.2E} ")
        print(f"Ensemble member {i} plotted")


    plt.tight_layout()
    save_as = f"{save_dir}/QBO_plots_seeds{plot_seeds}.png"
    plt.savefig(save_as, bbox_inches="tight")
    print(f"saved {save_as}")


