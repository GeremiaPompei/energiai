from torch import nn
import torch


class VAE(nn.Module):

    def __init__(self, features, timesteps, input_dim: int = 1, hidden_dim: int = 400, latent_dim: int = 200, device='cpu'):
        super(VAE, self).__init__()
        self.device = device

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * features * timesteps, latent_dim),
        ).to(torch.float64).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2).to(torch.float64).to(self.device)
        self.logvar_layer = nn.Linear(latent_dim, 2).to(torch.float64).to(self.device)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim * timesteps * features),
            nn.Unflatten(1, (latent_dim, timesteps, features)),
            nn.ConvTranspose2d(latent_dim, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Sigmoid()
        ).to(torch.float64).to(self.device)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
