from torch import nn
import torch


class VAE(nn.Module):

    def __init__(self, features, tau, input_dim: int = 1, hidden_dim: int = 400, latent_dim: int = 200, device='cpu'):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * features * tau, latent_dim),
        ).to(torch.float64)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2).to(torch.float64)
        self.logvar_layer = nn.Linear(latent_dim, 2).to(torch.float64)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim * tau * features),
            nn.Unflatten(1, (latent_dim, tau, features)),
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(torch.float64)

        self.device = device

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
        x = x.unsqueeze(dim=1)
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z).squeeze()
        return x_hat, mean, logvar
