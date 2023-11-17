from torch import nn
import torch


class VAE(nn.Module):

    def __init__(
            self,
            features,
            window: int = 10,
            hidden_dim: int = 200,
            latent_dim: int = 10,
            bias_perc_thresh: float = 0.2,
            device='cpu',
    ):
        super(VAE, self).__init__()
        self.device = device
        self.bias_perc_thresh = bias_perc_thresh
        self.tr_thresh = 0
        self.window = window

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(features, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * window, hidden_dim),
        ).to(torch.float64).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(hidden_dim, latent_dim).to(torch.float64).to(self.device)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim).to(torch.float64).to(self.device)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * window),
            nn.Unflatten(1, (hidden_dim, window)),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, features, kernel_size=3, padding=1),
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

    def mahalanobis_dist(self, z):
        z = z.abs()
        I = torch.eye(z.shape[-2]).to(self.device).to(z.dtype)
        return (z.transpose(-1, -2) @ I @ z).sqrt()

    def forward(self, x):
        x = x.unfold(1, self.window, 1).transpose(-1, -2)
        batch_size, features = x.shape[0], x.shape[-1]
        x = x.reshape(-1, self.window, features)
        if self.training:
            mean, logvar = self.encode(x.transpose(-1, -2))
            z = self.reparameterization(mean, logvar)
            self.tr_thresh = max(self.tr_thresh, self.mahalanobis_dist(z).max().item())
            x_hat = self.decode(z).transpose(-1, -2)
            return x, x_hat, mean, logvar
        else:
            threshold = (1 + self.bias_perc_thresh) * self.tr_thresh
            mean, logvar = self.encode(x.transpose(-1, -2))
            z = self.reparameterization(mean, logvar)
            dist = self.mahalanobis_dist(z.unsqueeze(-1)).squeeze()
            dist = (dist > threshold).to(torch.int64)
            res = torch.nn.functional.one_hot(dist, 3)
            res = res.reshape(batch_size, -1, 3)
            return res
