import torch
import torch.nn


class WeightsInitializer:

    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(
            self,
            shape: tuple,
            ratio: float = 1,
            sparsity: float = 0
    ) -> torch.Tensor:
        weights = (torch.rand(shape, generator=self.generator, requires_grad=False) * 2 - 1).to(torch.float64) * ratio
        weights[torch.rand(weights.shape, generator=self.generator) < sparsity] = 0
        return weights


class Reservoir(torch.nn.Module):

    def __init__(
            self,
            in_size: int,
            hyperparams: dict,
            initializer: WeightsInitializer,
            device: str = 'cpu'
    ):
        super(Reservoir, self).__init__()
        self.hyperparams = hyperparams
        self.layers = [
            ReservoirLayer(
                in_size if l == 0 else hyperparams['reservoir_size'],
                hyperparams,
                initializer,
                device
            ) for l in range(self.hyperparams['n_layers'])
        ]

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        states = []
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        states = torch.cat(states, dim=-1)
        return states


class ReservoirLayer(torch.nn.Module):

    def __init__(
            self,
            in_size: int,
            hyperparams: dict,
            initializer: WeightsInitializer,
            device: str = 'cpu'
    ):
        super(ReservoirLayer, self).__init__()
        self.hyperparams = hyperparams
        self.device = device

        self.W_in = initializer(
            (in_size, hyperparams['reservoir_size']),
            ratio=hyperparams['input_ratio'],
            sparsity=hyperparams['input_sparsity']
        ).to(device)

        self.W_hh = self.__init_reservoir(initializer).to(device)

        self.bias = initializer(
            (1, hyperparams['reservoir_size']),
            ratio=hyperparams['input_ratio'],
            sparsity=hyperparams['input_sparsity']
        ).to(device)

    def __init_reservoir(self, initializer: WeightsInitializer) -> torch.Tensor:
        weights = initializer(
            (self.hyperparams['reservoir_size'], self.hyperparams['reservoir_size']),
            sparsity=self.hyperparams['reservoir_sparsity']
        )
        max_eig = torch.linalg.eigvals(weights).abs().max()
        weights *= self.hyperparams['spectral_radius'] / max_eig
        return weights

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros((x.size(dim=1), self.W_hh.size(dim=0))).to(torch.float64).to(self.device)
        alpha = self.hyperparams['alpha']
        states = []
        for x_i in x:
            h = (1 - alpha) * h + alpha * torch.tanh(x_i @ self.W_in + h @ self.W_hh + self.bias)
            states.append(h)
        return torch.stack(states)


class ESN(torch.nn.Module):

    def __init__(
            self,
            in_size: int,
            device: str = 'cpu',
            window: int = 10,
            **hyperparams: dict,
    ):
        super(ESN, self).__init__()
        self.hyperparams = hyperparams
        self.window = window
        self.sigma = 0
        self.tr_data = 0
        self.device = device
        initializer = WeightsInitializer(hyperparams['seed'])

        self.reservoir = Reservoir(in_size, hyperparams, initializer, device)

        self.readout = initializer(
            (hyperparams['reservoir_size'] * hyperparams['n_layers'], in_size)
        ).to(device)

        self.A, self.B = None, None
        self.reset_AB()

    def __loss__(self, y, p):
        return ((y - p) ** 2).mean(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        h = self.reservoir(x)
        out = h @ self.readout
        out = out.transpose(0, 1)
        return out

    def register_std(self, x, y):
        out = self(x)
        e = self.__loss__(y, out)
        tot = self.tr_data + 1
        self.sigma = (self.sigma * self.tr_data + e.std()) / tot
        self.tr_data = tot

    def predict(self, x, y):
        out = self(x)
        e = self.__loss__(y, out)
        e_std = e.unfold(1, self.window, 1).std(-1)
        res = torch.logical_or(-2 * self.sigma > e_std, e_std > 2 * self.sigma).to(torch.float64)
        return res, out.detach(), e_std.detach()

    def reset_AB(self):
        self.A = None
        self.B = None

    def compute_ab(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        h = self.reservoir(x)
        washout = self.hyperparams['washout']
        h = h[washout:].reshape(-1, h.shape[-1])
        y = y[washout:].reshape(-1, y.shape[-1])
        A = h.T @ y
        B = h.T @ h
        return A, B

    def fit_from_ab(self, A: torch.Tensor, B: torch.Tensor):
        self.A, self.B = A, B
        self.readout = torch.linalg.pinv(
            self.B + self.hyperparams['regularization'] * torch.eye(self.B.shape[-1])) @ self.A

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        x, y = x.transpose(0, 1), y.transpose(0, 1)
        A, B = self.compute_ab(x, y)
        self.fit_from_ab(A=A, B=B)
