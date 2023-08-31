import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_QBO(time, pfull, u_QBO, ax=None, cbar=True, x_axis=True, y_axis=True):
    if ax == None:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8, 3))
    plt.sca(ax)
    plt.contourf(time, pfull, u_QBO.T, levels=np.arange(-45., 45.1, 5), cmap="RdBu_r", extend="both")
    if x_axis:
        plt.xlabel("Time (years)")
    if y_axis:
        plt.ylabel("Pressure (hPa)")
        ax.invert_yaxis()
        ax.set_yscale('log')
    if cbar:
        plt.colorbar()
    return ax

