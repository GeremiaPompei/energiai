from torch import nn
import torch


def basic_conv_layer(in_features, out_features):
    return nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)


def mobilenet_conv_layer(in_features, out_features):
    return nn.Sequential(
        nn.Conv1d(in_features, 1, kernel_size=3, padding=1),
        nn.Conv1d(1, out_features, kernel_size=1, padding=0),
    )


conv_layer_constructors = {c.__name__: c for c in [basic_conv_layer, mobilenet_conv_layer]}


class VAE(nn.Module):

    def __init__(
            self,
            features,
            timesteps,
            hidden_dim: int = 400,
            latent_dim: int = 200,
            conv_layer_consructor: str = 'basic_conv_layer',
            device='cpu',
    ):
        super(VAE, self).__init__()
        self.device = device
        conv_layer_consructor = conv_layer_constructors[conv_layer_consructor]

        # encoder
        self.encoder = nn.Sequential(
            conv_layer_consructor(features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            conv_layer_consructor(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * timesteps, latent_dim),
        ).to(torch.float64).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2).to(torch.float64).to(self.device)
        self.logvar_layer = nn.Linear(latent_dim, 2).to(torch.float64).to(self.device)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim * timesteps),
            nn.Unflatten(1, (latent_dim, timesteps)),
            conv_layer_consructor(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            conv_layer_consructor(hidden_dim, features),
            nn.BatchNorm1d(features),
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
