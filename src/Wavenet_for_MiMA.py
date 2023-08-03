import torch
import torch.nn as nn
import numpy as np

from utils import init_xavier


class Wavenet_for_MiMA(nn.Module):
    def __init__(self, n_d=[128,64,32,1], n_in=40, n_out=33, 
                 use_dropout=False,
                 transform_vars = {}
                 ):
        super(Wavenet_for_MiMA, self).__init__()
        self.n_in = n_in
        self.n_d = n_d
        self.n_out = n_out
        if use_dropout:
            dropout_rate = 0.5
        else:
            dropout_rate = 1
        self.shared = nn.Sequential(
            nn.BatchNorm1d(self.n_in),     # Added by Laura, not in orig WaveNet
            nn.Linear(self.n_in, self.n_d[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        self.branches = nn.ModuleList([ nn.Sequential(
            nn.Linear(self.n_d[0], self.n_d[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.n_d[1], self.n_d[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.n_d[2], self.n_d[3])
            ) for j in range(n_out)])
        self.shared.apply(init_xavier)
        for branch in self.branches:
            branch.apply(init_xavier)
            
        
        self.gwfu_mean = transform_vars["gwfu_mean"]
        self.u_mean = transform_vars["u_mean"]
        self.T_mean = transform_vars["T_mean"]
        self.ps_mean = transform_vars["ps_mean"]

        self.gwfu_sd = transform_vars["gwfu_sd"]
        self.u_sd = transform_vars["u_sd"]
        self.T_sd = transform_vars["T_sd"]
        self.ps_sd = transform_vars["ps_sd"]
    
    
    def forward(self, 
                u: torch.Tensor, 
                T: torch.Tensor, 
                lat: torch.Tensor, 
                ps: torch.Tensor, 
                lat_ind: torch.Tensor):
        
        lat_ind = lat_ind.to(torch.int64)   # should not need to do this
        ## Get X as scaled and concatenated variables
        # get means and sd
        u_mean = self.u_mean[0,:,lat_ind,0].squeeze(dim=2)
        u_sd = self.u_sd[0,:,lat_ind,0].squeeze(dim=2)
        T_mean = self.T_mean[0,:,lat_ind,0].squeeze(dim=2)
        T_sd = self.T_sd[0,:,lat_ind,0].squeeze(dim=2)
        ps_mean = self.ps_mean[0,0,lat_ind].squeeze(dim=2)
        ps_sd = self.ps_sd[0,0,lat_ind].squeeze(dim=2)

        # swap dims to correct size (batch, 40)
        u_mean = u_mean.transpose(0,1)
        u_sd = u_sd.transpose(0,1)
        T_mean = T_mean.transpose(0,1)
        T_sd = T_sd.transpose(0,1)

        # scale u,T,ps. Note lat is already in radians.
        u_scaled = self.standard_scaler(u, u_mean, u_sd)
        T_scaled = self.standard_scaler(T, T_mean, T_sd)
        ps_scaled = self.standard_scaler(ps, ps_mean, ps_sd)

        # concatenate into single torch tensor size (batch, 82)
        x = torch.concat( (u_scaled, 
                           T_scaled, 
                           lat, 
                           ps_scaled
                           ), dim=1)
        ## Predict gwfu
        gu = torch.zeros(x.shape[0], self.n_out, device=x.device)

        z = self.shared(x) 

        # Do branching.
        for j in range(self.n_out): gu[:,j]= self.branches[j](z).squeeze()
            
        ## Rescale and return gwfu 
        gwfu_mean = self.gwfu_mean[0,:,lat_ind,0].squeeze(dim=2).transpose(0,1)
        gwfu_sd = self.gwfu_sd[0,:,lat_ind,0].squeeze(dim=2).transpose(0,1)
        return self.inverse_standard_scaler(gu, gwfu_mean, gwfu_sd)

    
    def standard_scaler(self, u,  u_mean, u_sd):
        """Standard scaler can be applied to any variable, 
        you must have already done indexing i.e. u_mean = u_mean_global[0, :, lat_idx] 
        where lat_idx = list(np.where(self.lat==lat[i])[0])"""
        u = (u - u_mean) / u_sd
        u = torch.nan_to_num(u, nan=0)    # Remove nans that occur when sd = 0
        return u
    
    def inverse_standard_scaler(self, u, u_mean, u_sd):
        """Inverse Standard scaler can be applied to any variable, 
        you must have already done indexing i.e. u_mean = u_mean_global[0, :, lat_idx]
        where lat_idx = list(np.where(self.lat==lat[i])[0]) """
        u = u * u_sd + u_mean
        return u
