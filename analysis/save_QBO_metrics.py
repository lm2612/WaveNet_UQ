import xarray as xr
import numpy as np
import argparse
from get_QBO_TT_metrics import get_QBO_periods_amplitudes 

parser = argparse.ArgumentParser()
parser.add_argument('--seed', metavar='seed', type=int, nargs='+', default=1)
args = parser.parse_args()

base_dir = "/scratch/users/lauraman/WaveNetPyTorch/mima_runs"

for seed in args.seed:
    dir_name = f"wavenet_4yr_seed{seed}"
    filename = f"{base_dir}/{dir_name}/QBO_winds.nc"
    print(f"Opening file {filename}")
    ds = xr.open_dataset(filename, decode_times=False)
    QBO_winds = ds.ucomp.sel(pfull=10, method="nearest")
    print(QBO_winds[:100])
    periods, amplitudes = get_QBO_periods_amplitudes(QBO_winds, 5*30)

    print(periods, amplitudes)
