import torch
import torch.nn

from src.model.anomaly_detector import AnomalyDetector


class WeightsInitializer:

    def __init__(self, requires_grad: bool = False, seed: int = 0, dtype: torch.dtype = torch.float64):
        self.generator = torch.Generator().manual_seed(seed)
        self.requires_grad = requires_grad
        self.dtype = dtype

    def __call__(
            self,
            shape: tuple,
            ratio: float = 1,
            sparsity: float = 0
    ) -> torch.Tensor:
        weights = (torch.rand(shape, generator=self.generator, requires_grad=self.requires_grad) * 2 - 1).to(
            self.dtype) * ratio
        weights[torch.rand(weights.shape, generator=self.generator) < sparsity] = 0
        return weights


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
        )

        self.W_hh = self.__init_reservoir(initializer)

        self.bias = initializer(
            (1, hyperparams['reservoir_size']),
            ratio=hyperparams['input_ratio'],
            sparsity=hyperparams['input_sparsity']
        )

        params = [self.W_in, self.W_hh, self.bias]
        for i in range(len(params)):
            params[i] = params[i].to(device)
            if initializer.requires_grad:
                params[i] = torch.nn.Parameter(params[i])

    def __init_reservoir(self, initializer: WeightsInitializer) -> torch.Tensor:
        weights = initializer(
            (self.hyperparams['reservoir_size'], self.hyperparams['reservoir_size']),
            sparsity=self.hyperparams['reservoir_sparsity']
        )
        max_eig = torch.linalg.eigvals(weights).abs().max()
        weights.data *= self.hyperparams['spectral_radius'] / max_eig
        return weights

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros((x.size(dim=1), self.W_hh.size(dim=0))).to(x.dtype).to(self.device)
        alpha = self.hyperparams['alpha']
        states = torch.zeros(x.shape[0], h.shape[0], h.shape[1]).to(x.dtype).to(self.device)
        for i, x_i in enumerate(x):
            h = (1 - alpha) * h + alpha * torch.tanh(x_i @ self.W_in + h @ self.W_hh + self.bias)
            states[i, :, :] = h
        return states


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
        self.device = device
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
        f = self.hyperparams['reservoir_size']
        states = torch.ones((x.shape[0], x.shape[1], f * len(self.layers) + 1)).to(x.dtype).to(self.device)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            states[:, :, i * f:(i + 1) * f] = x
        return states


class ESN(AnomalyDetector):

    def __init__(
            self,
            in_size: int,
            device: str = 'cpu',
            **hyperparams: dict,
    ):
        super(ESN, self).__init__(**hyperparams)
        self.hyperparams = hyperparams
        self.device = device
        initializer = WeightsInitializer(**{
            k: hyperparams[k] for k in ['requires_grad', 'seed', 'dtype'] if k in hyperparams
        })

        self.reservoir = Reservoir(in_size, hyperparams, initializer, device)

        self.readout = initializer(
            (hyperparams['reservoir_size'] * hyperparams['n_layers'] + 1, in_size)
        ).to(device)
        if initializer.requires_grad:
            self.readout = torch.nn.Parameter(self.readout)

        self.A, self.B = None, None
        self.reset_AB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        h = self.reservoir(x)
        out = h @ self.readout
        out = out.transpose(0, 1)
        return out

    def reset_AB(self):
        self.A = None
        self.B = None

    def compute_ab(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        h = self.reservoir(x)
        washout = self.hyperparams['washout']
        h = h[:, washout:].reshape(-1, h.shape[-1])
        y = y[:, washout:].reshape(-1, y.shape[-1])
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
