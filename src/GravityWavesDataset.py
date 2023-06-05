import numpy as np
import xarray as xr

import torch
from torch.utils.data import Dataset

class GravityWavesDataset(Dataset):
    """Gravity wave AD99 dataset."""

    def __init__(self, data_dir, filename, npfull_out = 33, 
                 subset_time=None, transform=None, 
                 transform_dict = {},
                 ):
        """
        Sets up dataset from which you can index. Data must fit into memory, everything is
        done with numpy arrays. If not, use other Xarray/dask version, which is slower.
        Args:
            data_dir (string): Location of directory
            filename (string): Filename of netcdf containing ucomp, temp, gwfu_cgwd
            npfull_out (int): Number of outputs to predict
            subset_time (None or tuple): Either none or the slice of data to extract
                in time, e.g. if you want to select only 1 season or a smaller validation
                set. 
            transform (string or None, optional): Optional transform to be applied
                on a sample, either "minmax" or "standard" or None.
            transform_dict (dict, optional): Information needed for transform
                
        """
        path_to_file = data_dir + filename
        self.ds = xr.open_dataset(path_to_file, decode_times=False )
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

        print(f"Full dataset has ntime={self.ntime}")
        # Get variables 
        self.ucomp = self.ds["ucomp"]
        self.temp = self.ds["temp"]
        self.gwfu = self.ds["gwfu_cgwd"]
        if subset_time != None:
            time_start = subset_time[0]
            time_end = subset_time[1]
            # Update time xarray
            self.time = self.time[time_start:time_end]
            self.ntime = len(self.time)
            # Update variables
            self.ucomp = self.ucomp[time_start:time_end]
            self.temp = self.temp[time_start:time_end]
            self.gwfu = self.gwfu[time_start:time_end]
            print(f"Data is subset by time from {time_start} to {time_end}. ntime={self.ntime}")
        

        # Note we do not need pressure levels, give this a dummy 1D value
        dummy_val = np.array([np.nan])
        self.lon_expanded = self.lon.expand_dims(dim={'time':self.time, 
                                                      'pfull':dummy_val, 
                                                      'lat':self.lat}, 
                                       axis =(0,1,2))
        self.lat_expanded = self.lat.expand_dims(dim={'time':self.time, 
                                                      'pfull':dummy_val, 
                                                      'lon':self.lon}, 
                                       axis=(0,1,3))
        # Convert to numpy arrays (load into memory) for faster computation later
        self.ucomp = self.ucomp.to_numpy()
        self.temp = self.temp.to_numpy()
        self.gwfu = self.gwfu.to_numpy()
        self.lon = self.lon.to_numpy()
        self.lat = self.lat.to_numpy()
        self.time = self.time.to_numpy()
        self.pfull = self.pfull.to_numpy()
        self.lon_expanded = self.lon_expanded.to_numpy()
        self.lat_expanded = self.lat_expanded.to_numpy()

    
        # Other info needed for __get_item__()
        self.transform = transform
        self.transform_dict = transform_dict
        if transform:
            if "transform_dir" in transform_dict.keys():
                transform_dir =  transform_dict["transform_dir"]
            else:
                transform_dir =  data_dir
            ## Set up means and sd file for standard scaling. These
            ## must be saved in transform_dict
            if transform.lower() == "minmax":
                filename_min = transform_dict["filename_min"]
                with xr.open_dataset(transform_dir + filename_min, decode_times=False ) as ds_min:
                    self.gwfu_min = ds_min["gwfu_cgwd"].to_numpy() 
                    self.u_min = ds_min["ucomp"].to_numpy() 
                    self.T_min = ds_min["temp"].to_numpy() 
                
                filename_max = transform_dict["filename_max"]
                with xr.open_dataset(transform_dir + filename_max, decode_times=False ) as ds_max:
                    self.gwfu_max = ds_max["gwfu_cgwd"].to_numpy() 
                    self.u_max = ds_max["ucomp"].to_numpy() 
                    self.T_max = ds_max["temp"].to_numpy() 
            elif transform.lower() == "standard":
                filename_mean = transform_dict["filename_mean"]
                with xr.open_dataset(transform_dir + filename_mean, decode_times=False ) as ds_mean:
                    self.gwfu_mean = ds_mean["gwfu_cgwd"].to_numpy() 
                    self.u_mean = ds_mean["ucomp"].to_numpy() 
                    self.T_mean = ds_mean["temp"].to_numpy() 
                
                filename_sd = transform_dict["filename_sd"]
                with xr.open_dataset(transform_dir + filename_sd, decode_times=False ) as ds_sd:
                    self.gwfu_sd = ds_sd["gwfu_cgwd"].to_numpy() 
                    self.u_sd = ds_sd["ucomp"].to_numpy() 
                    self.T_sd = ds_sd["temp"].to_numpy() 
            else:
                print(f"Transform {transform} functionality does not exist")



    def __len__(self):
        #return len(self.ds['time']) * len(self.ds['lon']) * len(self.ds['lat'])
        return self.ntime * self.nlon * self.nlat


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

    def recover_index(self, time_ind, lat_ind, lon_ind):
        """ Recover single idx from time lat and lon inds"""
        return time_ind * (self.nlat*self.nlon) + lat_ind * self.nlon + lon_ind


    def minmax_scaler(self, u, T, gwfu, lat_idx, lon_idx):
        """Min max scaler"""
        gwfu_min = self.gwfu_min[0, :, lat_idx, lon_idx]
        u_min = self.u_min[0, :, lat_idx, lon_idx]
        T_min = self.T_min[0, :, lat_idx, lon_idx]
        gwfu_max = self.gwfu_max[0, :, lat_idx, lon_idx]
        u_max = self.u_max[0, :, lat_idx, lon_idx]
        T_max = self.T_max[0, :, lat_idx, lon_idx]

        # Apply min max scaler
        gwfu = (gwfu - gwfu_min) / (gwfu_max - gwfu_min)
        gwfu = np.nan_to_num(gwfu, nan=0)      # Remove nans that occur when min and max = 0
        u = (u - u_min) / (u_max - u_min)
        T = (T - T_min) / (T_max - T_min)
        return u, T, gwfu

    def inverse_minmax_scaler(self, u, T, gwfu, lat_idx, lon_idx):
        gwfu_min = self.gwfu_min[0, :self.npfull_out, lat_idx, lon_idx]
        u_min = self.u_min[0, :, lat_idx, lon_idx]
        T_min = self.T_min[0, :, lat_idx, lon_idx]
        gwfu_max = self.gwfu_max[0, :self.npfull_out, lat_idx, lon_idx]
        u_max = self.u_max[0, :, lat_idx, lon_idx]
        T_max = self.T_max[0, :, lat_idx, lon_idx]

        # Apply min max scaler
        gwfu = gwfu * (gwfu_max - gwfu_min) + gwfu_min
        u = u  * (u_max - u_min) +  u_min
        T = T * (T_max - T_min) + T_min
        return u, T, gwfu

    def standard_scaler(self, u, T, gwfu, lat_idx, lon_idx):
        """Standard scaler"""
        gwfu_mean = self.gwfu_mean[0, :, lat_idx, lon_idx]
        u_mean = self.u_mean[0, :, lat_idx, lon_idx]
        T_mean = self.T_mean[0, :, lat_idx, lon_idx]
        gwfu_sd = self.gwfu_sd[0, :, lat_idx, lon_idx]
        u_sd = self.u_sd[0, :, lat_idx, lon_idx]
        T_sd = self.T_sd[0, :, lat_idx, lon_idx]
        
        # Apply standard scaler
        zero_inds = np.where(gwfu_sd == 0)
        
        gwfu = (gwfu - gwfu_mean) / gwfu_sd
        gwfu = np.nan_to_num(gwfu, nan=0)      # Remove nans that occur when sd = 0
        u = (u - u_mean) / u_sd
        T = (T - T_mean) / T_sd
        return u, T, gwfu


    def inverse_standard_scaler(self, u, T, gwfu, lat_idx, lon_idx):
        """Standard scaler"""
        gwfu_mean = self.gwfu_mean[0, :self.npfull_out, lat_idx, lon_idx]
        u_mean = self.u_mean[0, :, lat_idx, lon_idx]
        T_mean = self.T_mean[0, :, lat_idx, lon_idx]
        gwfu_sd = self.gwfu_sd[0, :self.npfull_out, lat_idx, lon_idx]
        u_sd = self.u_sd[0, :, lat_idx, lon_idx]
        T_sd = self.T_sd[0, :, lat_idx, lon_idx]

        # Apply standard scaler
        gwfu = gwfu *  gwfu_sd + gwfu_mean
        u = u * u_sd + u_mean
        T = T * T_sd + T_mean
        return u, T, gwfu

    def get_from_3d_inds(self, time_idx, lat_idx, lon_idx, apply_transform=True):
        """Extracts X,Y from dataset given separate time, lat and lon index.
        Applies transform by default, but can be switched off to return raw X, Y 
        (e.g. for analysis/plotting).
        Returns X, Y."""
        # Select samples, these will have dimension (n_samples, n_pfull)
        # Use slicing as we want to keep structure of arrays
        time_idx = [time_idx]
        lat_idx = [lat_idx]
        lon_idx = [lon_idx]
        gwfu = self.gwfu[time_idx, :, lat_idx, lon_idx]
        u = self.ucomp[time_idx, :, lat_idx, lon_idx]
        T = self.temp[time_idx, :, lat_idx, lon_idx]
        lon_expanded = self.lon_expanded[time_idx, :, lat_idx, lon_idx]
        lat_expanded = self.lat_expanded[time_idx, :, lat_idx, lon_idx]
        
        ## if apply_transform is True and we have defined a transform to apply 
        if self.transform:
            if self.transform.lower() == "minmax":
                u, T, gwfu = self.minmax_scaler(u, T, gwfu, lat_idx, lon_idx)
            elif self.transform.lower() == "standard":
                u, T, gwfu = self.standard_scaler(u, T, gwfu, lat_idx, lon_idx)
            # Scale lat between -1 and +1
            lat_expanded = lat_expanded / 90.
            # Scale lon between -1 and +1
            lon_expanded = (lon_expanded-180.) / 180. 

        # Concatenate X and set up Y arrays
        #X = xr.concat((u, T, lat_expanded, lon_expanded), dim='pfull')
        X = np.concatenate((u, T, lat_expanded, lon_expanded), axis=-1 )

        Y = gwfu[:, :self.npfull_out]
        return X, Y


    def get_uT_from_X(self, X):
        u = X[:, :self.npfull_in]
        T = X[:, self.npfull_in:self.npfull_in*2]
        lat_expanded = X[:, self.npfull_in*2]
        lon_expanded = X[:, self.npfull_in*2 + 1]
        return (u, T, lat_expanded, lon_expanded)

    def inverse_scaler(self, X, Y):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if torch.is_tensor(Y):
            Y = Y.detach().numpy()

        u, T, lat, lon = self.get_uT_from_X(X)
        gwfu = Y

        n_batch = lat.shape[0]
        
        lat_scaled = self.lat / 90.
        lon_scaled = (self.lon - 180.) / 180. 

        for i in range(n_batch):
            ## Extract lat and lon indices
            lat_idx = list(np.where(lat_scaled==lat[i])[0])
            lon_idx = list(np.where(lon_scaled==lon[i])[0])

            if self.transform.lower() == "minmax":
                ui, Ti, gwfui = self.inverse_minmax_scaler(u[i], T[i], gwfu[i],
                                                        lat_idx, lon_idx)
            elif self.transform.lower() == "standard":
                ui, Ti, gwfui = self.inverse_standard_scaler(u[i], T[i], gwfu[i],
                                                          lat_idx, lon_idx)

            ## Concatenate to put back into X, Y format, including lat/lon in xarray form
            lon_expanded = self.lon_expanded[0, :, lat_idx, lon_idx]
            lat_expanded = self.lat_expanded[0, :, lat_idx, lon_idx]
            X[i,:] = np.concatenate((ui, Ti, lat_expanded, lon_expanded), axis=-1)
            Y[i,:] = gwfui

        return X, Y



    def __getitem__(self, idx):
        """Generates one sample at index idx (int)"""
        # find time, lat and lon indices for idx
        time_idx, lat_idx, lon_idx = self.get_time_lat_lon(idx)

        X, Y = self.get_from_3d_inds(time_idx, lat_idx, lon_idx,
                                       apply_transform=True)
        
        # Return as torch tensors (xarray -> numpy -> torch)
        sample = {'X': torch.from_numpy(X),
                  'Y': torch.from_numpy(Y)}

        return sample
