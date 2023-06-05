import torch
import torch.nn as nn

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
                   
class VAE(nn.Module):
    def __init__(self, n_in=40, n_out=33, latent_dim=4):
        super(VAE, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.latent_dim = latent_dim
  
        self.encoder = nn.Sequential(
            nn.Linear(self.n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
            nn.ReLU()
        )
        self.encoder.apply(init_xavier)
        self.decoder.apply(init_xavier)

    
    def encoder_block(self):
        # encoder x -> z
        # vaguely based on wavenet, no convolutions
        encoder = nn.Sequential(
            nn.Linear(self.n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_dim),
            nn.ReLU()
        )
        return encoder
        
    def decoder_block(self):
        # decoder z -> x
        # latent dim is size 4
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_out),
            nn.ReLU()
        )
        return decoder
        
    
    def forward(self, x):
        z = self.encoder(x)      # z is size (N, 4)
        return self.decoder(z)
        
