import numpy as np
import xarray as xr
import dask

import torch
from torch.utils.data import Dataset


class GravityWavesDataset(Dataset):
    """Gravity wave AD99 dataset."""

    def __init__(self, data_dir, filename, npfull_out = 40, 
                 subset_time=None, transform=None, 
                 transform_dict = {}, component="zonal",
                 ):
        """
        Sets up dataset from which you can index. Data must fit into memory, everything is
        done with numpy arrays. If not, use other Xarray/dask version, which is slower.
        Args:
            data_dir (string): Location of directory
            filename (string) or Tuple/List of strings: Filename of netcdf containing ucomp, temp, gwfu_cgwd
            npfull_out (int): Number of outputs to predict
            subset_time (None or tuple): Either none or the slice of data to extract
                in time, e.g. if you want to select only 1 season or a smaller validation
                set. 
            transform (string or None, optional): Optional transform to be applied
                on a sample, either "minmax" or "standard" or None.
            transform_dict (dict, optional): Information needed for transform
            component (string, optional): Either zonal or meridional.
                
        """
        if isinstance(filename, str):
            filename0 = filename
            print(f"Setting up dataset from single file: {filename0}")
        else:
            filename0 = filename[0]
            print(f"Setting up dataset from multiple files. Starting with {filename0}")

        path_to_file = data_dir + filename0
        self.ds = xr.open_dataset(path_to_file, decode_times=False , chunks={})
        if subset_time != None:
            self.ds = self.ds.isel(time=slice(time_start,time_end))

        # Get variables 
        if component.lower() == "zonal":
            wind_comp = "ucomp"
            gwf_comp = "gwfu_cgwd"
        elif component.lower() == "meridional":
            wind_comp = "vcomp"
            gwf_comp = "gwfv_cgwd"

        if isinstance(filename, str):
            print(f"Done opening single file")
        else:
            print(f"Done opening first file, now opening rest and concatenating")
            n_files = len(filename)
            print(f"Number of files: {n_files}")
            for file_no in range(1, n_files):
                filename_n = filename[file_no]
                print(f"Setting up {filename_n}")
                path_to_file = data_dir + filename_n
                ds_n = xr.open_dataset(path_to_file, decode_times=False, chunks={})
                if subset_time != None:
                    ds_n = ds_n.isel(time=slice(time_start,time_end))
                self.ds = xr.concat((self.ds, ds_n), dim="time", data_vars=[gwf_comp, wind_comp, "temp", "ps"] )
                
            print(f"Done opening and concatenating all files into one xarray dataset.")

 
        # Get dimensions
        self.lon = self.ds["lon"]
        self.lat = self.ds["lat"]
        self.time = self.ds["time"]
        self.pfull = self.ds["pfull"]

        self.ntime = len(self.time)
        self.npfull_in = len(self.pfull)
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        self.npfull_out = npfull_out

        print(f"Dataset has ntime={self.ntime}. Getting variables.")
        # Get variables 
        if component.lower() == "zonal":
            wind_comp = "ucomp"
            gwf_comp = "gwfu_cgwd"
        elif component.lower() == "meridional":
            wind_comp = "vcomp"
            gwf_comp = "gwfv_cgwd"

        self.gwfu = self.ds[gwf_comp]
        self.ucomp = self.ds[wind_comp]
        self.temp = self.ds["temp"]
        self.ps = self.ds["ps"]

        # Note we do not need pressure levels, give this a dummy 1D value
        self.dummy_val = np.array([np.nan])
        
        self.lat_expanded = self.lat.expand_dims(dim={'time':self.time, 
                                                      'pfull':self.dummy_val, 
                                                      'lon':self.lon}, 
                                       axis=(0,1,3))
        
        self.ps_expanded = self.ps.expand_dims(dim={'pfull':self.dummy_val}, 
                                       axis=(1))
        
        # Convert to numpy arrays (load into memory) for faster computation later
        print("Load into memory")
        self.ucomp = self.ucomp.to_numpy()
        self.temp = self.temp.to_numpy()
        self.gwfu = self.gwfu.to_numpy()
        self.lon = self.lon.to_numpy()
        self.lat = self.lat.to_numpy()
        self.ps = self.ps.to_numpy()
        self.time = self.time.to_numpy()
        self.pfull = self.pfull.to_numpy()
        self.lat_expanded = self.lat_expanded.to_numpy()
        self.ps_expanded = self.ps_expanded.to_numpy()
        print("Done. All variables as numpy arrays")

    
        # Other info needed for __get_item__()
        self.transform_on = False
        self.transform = transform
        self.transform_dict = transform_dict
        if transform:
            print("Setting up transforms")
            if "transform_dir" in transform_dict.keys():
                transform_dir =  transform_dict["transform_dir"]
            else:
                transform_dir =  data_dir
            ## Set up means and sd file for standard scaling. These
            ## must be saved in transform_dict
            if transform.lower() == "minmax":
                filename_min = transform_dict["filename_min"]
                with xr.open_dataset(transform_dir + filename_min, decode_times=False ) as ds_min:
                    self.gwfu_min = ds_min[gwf_comp].to_numpy().mean(axis=(3), keepdims=True) 
                    self.u_min = ds_min[wind_comp].to_numpy().mean(axis=(3), keepdims=True) 
                    self.T_min = ds_min["temp"].to_numpy().mean(axis=(3), keepdims=True) 
                    self.ps_min = ds_min["ps"].expand_dims(dim={'pfull':self.dummy_val},
                                                           axis=(1)).to_numpy().mean(axis=(3), keepdims=True)
                
                filename_max = transform_dict["filename_max"]
                with xr.open_dataset(transform_dir + filename_max, decode_times=False ) as ds_max:
                    self.gwfu_max = ds_max[gwf_comp].to_numpy().mean(axis=(3), keepdims=True) 
                    self.u_max = ds_max[wind_comp].to_numpy().mean(axis=(3), keepdims=True) 
                    self.T_max = ds_max["temp"].to_numpy().mean(axis=(3), keepdims=True) 
                    self.ps_max = ds_max["ps"].expand_dims(dim={'pfull':self.dummy_val},
                                                           axis=(1)).to_numpy().mean(axis=(3), keepdims=True)
                    
                ## Apply transform so that dataset returns transformed variables only
                self.apply_minmax_scaler()
                
            elif transform.lower() == "standard":
                filename_mean = transform_dict["filename_mean"]
                with xr.open_dataset(transform_dir + filename_mean, decode_times=False ) as ds_mean:
                    self.gwfu_mean = ds_mean[gwf_comp].to_numpy().mean(axis=(3), keepdims=True)
                    self.u_mean = ds_mean[wind_comp].to_numpy().mean(axis=(3), keepdims=True)
                    self.T_mean = ds_mean["temp"].to_numpy().mean(axis=(3), keepdims=True)
                    self.ps_mean = ds_mean["ps"].expand_dims(dim={'pfull':self.dummy_val},
                                                           axis=(1)).to_numpy().mean(axis=(3), keepdims=True)
                
                
                filename_sd = transform_dict["filename_sd"]
                with xr.open_dataset(transform_dir + filename_sd, decode_times=False ) as ds_sd:
                    self.gwfu_sd = ds_sd[gwf_comp].to_numpy().mean(axis=(3), keepdims=True)
                    self.u_sd = ds_sd[wind_comp].to_numpy().mean(axis=(3), keepdims=True)
                    self.T_sd = ds_sd["temp"].to_numpy().mean(axis=(3), keepdims=True)
                    self.ps_sd = ds_mean["ps"].expand_dims(dim={'pfull':self.dummy_val},
                                                           axis=(1)).to_numpy().mean(axis=(3), keepdims=True)
                    
                ## Apply transform so that dataset returns transformed variables only. 
                self.apply_standard_scaler()                

            else:
                print(f"Transform {transform} functionality does not exist")
        
        print("Done. Dataset ready.")



    def __len__(self):
        return self.ntime * self.nlon * self.nlat


    def __getitem__(self, idx):
        """Generates one sample at index idx (int)"""
        # find time, lat and lon indices for idx
        time_idx, lat_idx, lon_idx = self.get_time_lat_lon(idx)

        X, Y = self.get_from_3d_inds(time_idx, lat_idx, lon_idx)
        # Return as torch tensors 
        sample = {'X': torch.from_numpy(X),
                  'Y': torch.from_numpy(Y)}
        return sample
    
    def get_time_lat_lon(self, idx):
        """Extracts unique time, lat and lon index from a single integer index """
        ## assume list of length (ntotal) is split into segments as follows
        ## | TIME = 0 [[ LON = 0, LAT = 0,1,... | LON = 1, LAT = 0,1,... | .... | ]] | 
        ## | TIME = 1 [[ LON = 0, LAT = 0,1,... | LON = 1, LAT = 0,1... | ... | ]] | 
        ## | .....
        # [ (0 0 0) , (0 0 1) , (0 0 2),...(0 0 NLON),(0 1 0),...(0 NLAT NLON), (1 0 0)... ]
        nspace = self.nlat * self.nlon
        ## Get time index first
        time_ind = idx // nspace
        ## in the first time index
        lonlat_ind = idx % nspace
        ## Next lat index
        lat_ind = lonlat_ind // self.nlon
        lon_ind = lonlat_ind % self.nlon
        return(time_ind, lat_ind, lon_ind)
    
    def get_from_3d_inds(self, time_idx, lat_idx, lon_idx):
        """Extracts X,Y from dataset given separate time, lat and lon index.
        Returns X, Y."""
        # Select samples, these will have dimension (n_samples, n_pfull)
        # Use slicing as we want to keep structure of arrays
        time_idx = [time_idx]
        lat_idx = [lat_idx]
        lon_idx = [lon_idx]
        gwfu = self.gwfu[time_idx, :, lat_idx, lon_idx]
        u = self.ucomp[time_idx, :, lat_idx, lon_idx]
        T = self.temp[time_idx, :, lat_idx, lon_idx]
        lat_expanded = self.lat_expanded[time_idx, :, lat_idx, lon_idx]
        ps_expanded = self.ps_expanded[time_idx, :, lat_idx, lon_idx]

        # Concatenate X and set up Y arrays
        X = np.concatenate((u, T, lat_expanded, ps_expanded), axis=-1 )
        Y = gwfu[:, :self.npfull_out]
        return X, Y

    def get_uTlp_from_X(self, X):
        u = X[:, :self.npfull_in]
        T = X[:, self.npfull_in:self.npfull_in*2]
        lat_expanded = X[:, self.npfull_in*2]
        ps_expanded = X[:, self.npfull_in*2 + 1]
        return (u, T, lat_expanded, ps_expanded)

    def recover_index(self, time_ind, lat_ind, lon_ind):
        """ Recover single idx from time lat and lon inds"""
        return time_ind * (self.nlat*self.nlon) + lat_ind * self.nlon + lon_ind


    def minmax_scaler(self, u, u_min, u_max):
        """Min max scaler can be applied to any variable, 
        you must have already done indexing i.e. u_min = u_min_global[0, :, lat_idx, lon_idx]"""
        u = (u - u_min) / (u_max - u_min)
        u = np.nan_to_num(u, nan=0)    # Remove nans that occur when min and max = 0
        return u
    
    def inverse_minmax_scaler(self, u, u_min, u_max):
        """Inverse min max scaler can be applied to any variable, 
        you must have already done indexing i.e. u_min = u_min_global[0, :, lat_idx, lon_idx]"""
        u = u  * (u_max - u_min) +  u_min
        return u
    
    def apply_minmax_scaler(self):
        if self.transform_on:
            print("Transform already applied to this dataset, doing nothing)")
            return
        else:
            self.ucomp = self.minmax_scaler(self.ucomp, self.u_min, self.u_max)
            self.gwfu = self.minmax_scaler(self.gwfu, self.gwfu_min, self.gwfu_max)
            self.temp = self.minmax_scaler(self.temp, self.T_min, self.T_max)
            self.ps_expanded = self.minmax_scaler(self.ps_expanded, self.ps_min, self.ps_max)
            self.lat_expanded = self.lat_expanded * np.pi / 180. 
            self.transform_on = True
            print("Dataset will return transformed variables (minmax scaler)")

    def apply_inverse_minmax_scaler(self):
        if self.transform_on==False:
            print("Transform not applied to this dataset, doing nothing")
            return
        else: 
            self.ucomp = self.inverse_minmax_scaler(self.ucomp, self.u_min, self.u_max)
            self.gwfu = self.inverse_minmax_scaler(self.gwfu, self.gwfu_min, self.gwfu_max)
            self.temp = self.inverse_minmax_scaler(self.temp, self.T_min, self.T_max)
            self.ps_expanded = self.inverse_minmax_scaler(self.ps_expanded, self.ps_min, self.ps_max)
            self.lat_expanded = self.lat_expanded *  180.  / np.pi 
            self.transform_on = False
        print("Dataset will return raw variables")
        

    def standard_scaler(self, u,  u_mean, u_sd):
        """Standard scaler can be applied to any variable, 
        you must have already done indexing i.e. u_mean = u_mean_global[0, :, lat_idx] 
        where lat_idx = list(np.where(self.lat==lat[i])[0])"""
        u = (u - u_mean) / u_sd
        u = np.nan_to_num(u, nan=0)    # Remove nans that occur when sd = 0
        return u

    def inverse_standard_scaler(self, u, u_mean, u_sd):
        """Inverse Standard scaler can be applied to any variable, 
        you must have already done indexing i.e. u_mean = u_mean_global[0, :, lat_idx]
        where lat_idx = list(np.where(self.lat==lat[i])[0]) """
        u = u * u_sd + u_mean
        return u
    
    def apply_standard_scaler(self):
        if self.transform_on:
            print("Transform already applied to this dataset, doing nothing)")
            return
        else:
            
            self.ucomp = self.standard_scaler(self.ucomp, self.u_mean, self.u_sd)
            self.gwfu = self.standard_scaler(self.gwfu, self.gwfu_mean, self.gwfu_sd)
            self.temp = self.standard_scaler(self.temp, self.T_mean, self.T_sd)
            self.ps_expanded = self.standard_scaler(self.ps_expanded, self.ps_mean, self.ps_sd)
            self.lat_expanded = self.lat_expanded * np.pi / 180. 
            self.transform_on = True
            print("Dataset will return transformed variables (standard scaler)")

    def apply_inverse_standard_scaler(self):
        if self.transform_on==False:
            print("Transform not applied to this dataset, doing nothing")
            return
        else: 
            self.ucomp = self.inverse_standard_scaler(self.ucomp, self.u_mean, self.u_sd)
            self.gwfu = self.inverse_standard_scaler(self.gwfu, self.gwfu_mean, self.gwfu_sd)
            self.temp = self.inverse_standard_scaler(self.temp, self.T_mean, self.T_sd)
            self.ps_expanded = self.inverse_standard_scaler(self.ps_expanded, self.ps_mean, self.ps_sd)
            self.lat_expanded = self.lat_expanded *  180.  / np.pi 
            self.transform_on = False
        print("Dataset will return raw variables")

    
