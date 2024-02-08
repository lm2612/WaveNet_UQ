import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

## Plot QBO and show training, validation and test data (Fig. 1)
# Directories
data_dir="/scratch/users/lauraman/WaveNetPyTorch/mima_runs/train_wavenet/"
save_dir="/scratch/users/lauraman/WaveNetPyTorch/mima_runs/PLOTS/"

years = range(43, 55)

ds = xr.open_dataset(f"{data_dir}atmos_daily_{years[0]}.nc", decode_times=False)

# Dimensions
lat = ds["lat"]
lon = ds["lon"]
pfull = ds["pfull"]
time = ds["time"]

nlat = len(lat)
nlon = len(lon)
npfull = len(pfull)

# Variables
u = ds["ucomp"]
gwfu = ds["gwfu_cgwd"]

# Get QBO
uQBO = u.sel(lat=slice(-5,5)).mean(dim=("lat","lon"))
gwfuQBO = gwfu.sel(lat=slice(-5,5)).mean(dim=("lat","lon"))

## Concatenate
for t in years[1:]:
    ds = xr.open_dataset(f"{rundir}atmos_daily_{t}.nc", decode_times=False)
    uQBO = xr.concat((uQBO, ds["ucomp"].sel(lat=slice(-5,5)).mean(dim=("lat","lon"))), dim="time")
    gwfuQBO = xr.concat((gwfuQBO, ds["gwfu_cgwd"].sel(lat=slice(-5,5)).mean(dim=("lat","lon"))), dim="time")

    time = xr.concat((time, ds["time"]), dim="time")

nyears = years[-1] - years[0]
print(nyears)
train_years = 1
valid_years = 1

# Show up to 12 years of this dataset in total (1 year train, 1 year valid, 10 year test)
uQBO = uQBO[:12*360]
time = time[:12*360]
gwfuQBO = gwfuQBO[:12*360]

## Plot QBO
matplotlib.rc('font', **{"size":14})

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
plt.sca(axs[0])
plt.contourf((time-time[0])/360., pfull, uQBO.T, 
             levels=np.arange(-45., 45.1, 5), 
             cmap="BrBG_r", extend="both")
plt.ylabel("Pressure (hPa)")
plt.axis(ymin=1, ymax=1e2)
plt.gca().invert_yaxis()
plt.gca().set_yscale('log')

## Add train, valid and test labels
plt.axvline(x=1, color="black", linestyle="--", lw=3)
plt.annotate('Train', xy=(0.49, 0.9), xytext=(0.49, 6e-1),
             color="black", fontsize=16, 
             ha='center', va='bottom',
             zorder=9,
             annotation_clip=False,
            arrowprops=dict(arrowstyle='-[, widthB=1.72, lengthB=1.5', 
                            lw=2.0, color='black'))
plt.axvline(x=2, color="black", linestyle="--", lw=3)
plt.annotate('Valid', xy=(1.52, 0.9), xytext=(1.52, 6e-1), 
             color="red", fontsize=16, 
             ha='center', va='bottom',
             annotation_clip=False,
             zorder=12,
             arrowprops=dict(arrowstyle='-[, widthB=1.72, lengthB=1.5', 
                            lw=2.0, color='red'))
plt.annotate('Test', xy=(7.03, 0.9), xytext=(7.03, 6e-1),
            fontsize=16, ha='center', va='bottom', annotation_clip=False,
             zorder=8,
            arrowprops=dict(arrowstyle='-[, widthB=17.363, lengthB=1.5', 
                            lw=2.0, color='k'))

## Add a) labels
plt.text(x=-1.5, y=0.7, s="a)", fontsize=20)

## Add colorbar
axins = inset_axes(axs[0],
                    width="3%",  
                    height="100%",
                    loc='center right',
                    borderpad=-5
                   )
plt.colorbar( cax=axins, orientation="vertical",label="u m/s")


plt.sca(axs[1])
plt.contourf((time-time[0])/360., pfull, gwfuQBO.T, 
             levels=np.arange(-1.1e-5, 1.15e-5, 1e-6), 
             cmap="BrBG_r", extend="both")
plt.xlabel("Time (years)")
plt.ylabel("Pressure (hPa)")
plt.axis(ymin=1, ymax=1e2)
plt.gca().invert_yaxis()
plt.gca().set_yscale('log')
plt.axvline(x=1, color="black", linestyle="--", lw=3)
plt.axvline(x=2, color="black", linestyle="--", lw=3)
plt.title("")

## Add b) label
plt.text(x=-1.5, y=0.7, s="b)", fontsize=20)

## Add colorbar
axins = inset_axes(axs[1],
                    width="3%",  
                    height="100%",
                    loc='center right',
                    borderpad=-5
                   )
plt.colorbar( cax=axins, orientation="vertical",
             ticks=np.arange(-1e-5, 1.1e-5, 5e-6),
             label="GWD m/s^2")

save_as = f"{save_dir}/training_data_QBO.png"
plt.savefig(save_as, bbox_inches="tight")

