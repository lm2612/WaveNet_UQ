import torch
import torch.nn as nn

from utils import init_xavier

class Wavenet(nn.Module):
    def __init__(self, n_d=[128,64,32,1], n_in=40, n_out=33, n_ch=1, dropout_rate=0):
        super(Wavenet, self).__init__()
        self.n_in = n_in
        self.n_d = n_d
        self.n_out = n_out
        self.n_ch = n_ch
        self.shared = nn.Sequential(
            #nn.BatchNorm1d(self.n_ch),     # Added by Laura, not in orig WaveNet
            nn.Linear(self.n_in, self.n_d[0]),       
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            #nn.BatchNorm1d(self.n_d[0]) # ADDED
            )
        self.branches = nn.ModuleList([ nn.Sequential(
            nn.Linear(self.n_d[0], self.n_d[1]),       
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            #nn.BatchNorm1d(self.n_d[1]), ## ADDED
            nn.Linear(self.n_d[1], self.n_d[2]),       
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            #nn.BatchNorm1d(self.n_d[2]), ## ADDED
            nn.Linear(self.n_d[2], self.n_d[3])
            ) for j in range(n_out)])
        self.shared.apply(init_xavier)
        for branch in self.branches:
            branch.apply(init_xavier)
    def forward(self, x):
        gu = torch.zeros((x.shape[0], self.n_d[3], self.n_out), device=x.device)
        z = self.shared(x) 
        # Do branching.
        for j in range(self.n_out): 
            gu[..., j] = self.branches[j](z)
        return gu
