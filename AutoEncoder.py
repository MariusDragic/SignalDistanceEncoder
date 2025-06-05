# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class AutoEncoder(nn.Module):
    """
    Class that defines the architecture of the AutoEncoder model.
    2 possible architectures:
    - Fully connected autoencoder with 6 hidden layers
    - Convolutional autoencoder with 5 convolutional layers
    """

    def __init__(self, input_dim, latent_dim, conv=False):
        super(AutoEncoder, self).__init__()
        self.conv = conv
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if self.conv:

            self.encoder = nn.Sequential(
            
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2,padding=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

            self.last_time = input_dim // 32
            self.flat_size = 256 * self.last_time
            self.fc_enc = nn.Linear(self.flat_size, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, self.flat_size)

            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 128, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 1, 5, stride=2, padding=2, output_padding=1)
            )
        else:

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(1024, 512),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(512, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(256, 512),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(512, 1024),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(1024, input_dim)
            )
    def forward(self, x):
        if self.conv:

            if x.dim() == 2:  
                x = x.unsqueeze(1)
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)

            h = self.encoder(x)
            h = h.view(h.size(0), -1) 
            z = self.fc_enc(h) 

            h = self.fc_dec(z)
            h = h.view(h.size(0), 256, self.last_time)

            x_recon = self.decoder(h)

            return x_recon.squeeze(1), z

        else:
            z = self.encoder(x)
            x_recon = self.decoder(z)
            
            return x_recon, z
