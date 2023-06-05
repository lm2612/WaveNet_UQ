import torch
import torch.nn as nn


def init_xavier(m):
    if (type(m) == nn.Conv2d) or (
        type(m) == nn.Linear) or (
        type(m) == nn.ConvTranspose2d) or (
        type(m) == nn.Conv1d) or (
        type(m) == nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight)


class ConvNet(nn.Module):
    def __init__(self, n_in=40, n_out=33):
        super(ConvNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)
        ## Convolutional NN                         SIZE (NBATCH, 2, 40)
        self.main = nn.Sequential(
            #nn.BatchNorm1d(self.n_in),  
            nn.Conv1d(2, 2, 2, stride=1, padding=1),     #(NBATCH, 2, 41)
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),        #(NBATCH, 2, 21)
            nn.Conv1d(2, 2, 2, stride=1, padding=1),     #(NBATCH, 2, 22)
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),        #(NBATCH, 2, 12)
            nn.Conv1d(2, 2, 2, stride=1, padding=1),     #(NBATCH, 2, 13)
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),        #(NBATCH, 2, 7)
            nn.Flatten(),                                #(NBATCH,  14)
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 64),                        
            nn.ReLU(),
            nn.Linear(64, n_out),                        #(NBATCH,  40)
            nn.ReLU()
            )
        self.main.apply(init_xavier)
    
        ## Add max pooling
        ## flatten
        ## fully connected NN
    
    def forward(self, x):
        if x.ndim == 2:
            # Add the channel dimension
            x = x.unsqueeze(1)
        return self.main(x)
        
