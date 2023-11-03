from torch import nn
import torch


class VAE(nn.Module):

    def __init__(
            self,
            features,
            hidden_dim: int = 400,
            latent_dim: int = 200,
            n_layers: int = 0,
            device='cpu',
    ):
        super(VAE, self).__init__()
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ) for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, latent_dim),
        ).to(torch.float64).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, latent_dim).to(torch.float64).to(self.device)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim).to(torch.float64).to(self.device)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ) for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, features),
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
