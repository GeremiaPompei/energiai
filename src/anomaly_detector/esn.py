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
        weights = (torch.rand(shape, generator=self.generator, requires_grad=False) * 2 - 1) * ratio
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
        h = torch.zeros((x.size(dim=1), self.W_hh.size(dim=0))).to(self.device)
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
            out_size: int,
            hyperparams: dict,
            many_to_one: bool,
            device: str = 'cpu',
    ):
        super(ESN, self).__init__()
        self.hyperparams = hyperparams
        self.many_to_one = many_to_one
        self.device = device
        initializer = WeightsInitializer(hyperparams['seed'])

        self.reservoir = Reservoir(in_size, hyperparams, initializer, device)

        self.readout = initializer(
            (hyperparams['reservoir_size'] * hyperparams['n_layers'], out_size)
        ).to(device)

        self.A, self.B = None, None
        self.reset_AB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reservoir(x)
        if self.many_to_one:
            h = h[-1]
        return h @ self.readout

    def reset_AB(self):
        reservoir_size = self.hyperparams['reservoir_size']
        out_size = self.readout.size(dim=1)
        self.A = torch.zeros(
            (reservoir_size, out_size),
            dtype=torch.float
        ).to(self.device)
        self.B = torch.zeros(
            (reservoir_size, reservoir_size),
            dtype=torch.float
        ).to(self.device)

    def compute_ab(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        h = self.reservoir(x)
        if not self.many_to_one:
            washout = self.hyperparams['washout']
            h = h[washout:].reshape(-1, h.shape[-1])
            y = y[washout:].reshape(-1, y.shape[-1])
        else:
            h = h[-1]
        A = h.T @ y
        B = h.T @ h
        return A, B

    def fit_from_ab(self, A: torch.Tensor, B: torch.Tensor):
        self.A, self.B = A, B
        self.readout = torch.linalg.pinv(
            self.B + self.hyperparams['regularization'] * torch.eye(self.B.shape[-1])) @ self.A

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        A, B = self.compute_ab(x, y)
        self.fit_from_ab(A=A, B=B)