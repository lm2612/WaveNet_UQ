import torch
import torch.nn as nn

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
                   

class Wavenet(nn.Module):
    def __init__(self, n_d=[128,64,32,1], n_in=40, n_out=33, use_dropout=False, dropout_rate=0.5):
        super(Wavenet, self).__init__()
        self.n_in = n_in
        self.n_d = n_d
        self.n_out = n_out
        if use_dropout:
            dropout_rate = dropout_rate
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
    def forward(self, x):
        ## IF x contains the input (x3), as well as the memory allocation for the output (gu), then do the following. 
        ##x3, gu = x
        ## ELSE
        x3 = x
        gu = torch.zeros(x3.shape[0], self.n_out, device=x3.device)

        # Reshape x3 by concatenating the two fields (U, T).
        ##x3 = torch.reshape(x3, (x3.shape[0], x3.shape[1]*x3.shape[2]))

        z = self.shared(x3) #; print(z3.shape)

        # Do branching.
        for j in range(self.n_out): gu[:,j]= self.branches[j](z).squeeze()
        return gu
