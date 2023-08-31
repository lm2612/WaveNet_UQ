import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from mean_lat_weighted import mean_lat_weighted
from plot_QBO import plot_QBO

base_dir = "/scratch/users/lauraman/MiMA_pytorch/runs"
dir_names = ["AD99_run", "ML_run_2yr_scaling_nd", "ML_run_4yr_scaling_nd"] #, "ML_run_8yr_scaling_nd"]
names = ["AD99", "ML: 2yr training", "ML: 4 yr training"] #, "ML: 8 yr training"]

save_dir = f"{base_dir}/PLOTS"
save_plot = f"{save_dir}/compare_QBO_nyear.png"
N = len(dir_names)
n_year = 6      # number of years to plot
fig, axs = plt.subplots(nrows=N,ncols=1, figsize=(2*n_year, 3*N), sharex = True)

for dir_name, name, ax in zip(dir_names, names, axs):
    print(f"Opening file for {dir_name}")
    ds = xr.open_dataset(f"{base_dir}/{dir_name}/QBO_winds.nc", decode_times=False)
    u_QBO = ds["ucomp"]
    pfull = ds["pfull"]
    time = ds["time"]
    plot_QBO((time-time[0])/360., pfull, u_QBO, ax, cbar=False, x_axis=False)
    plt.title(name)

    
# On lower most axis add time axis and colorbar
plt.xlabel("Time (years)")
axins = inset_axes(ax,
                    width="100%",  
                    height="10%",
                    loc='lower center',
                    borderpad=-6
                   )
plt.colorbar(cax=axins, orientation="horizontal")
plt.savefig(save_plot)
print(f"QBO figure saved as {save_plot}")
