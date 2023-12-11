import torch
import torch.nn.functional as F

from src.model.anomaly_detector import AnomalyDetector


class LSTM(AnomalyDetector):
    def __init__(
            self,
            input_size,
            hidden_state: int = 100,
            n_layers: int = 1,
            bidirectional: bool = False,
            dropout: float = 0,
            device='cpu',
            **hyperparams: dict,
    ):
        super(LSTM, self).__init__(**hyperparams)
        self.lstm = torch.nn.LSTM(input_size, hidden_state, num_layers=n_layers, bidirectional=bidirectional,
                                  batch_first=True, dropout=dropout).to(torch.float64).to(device)
        self.ff = torch.nn.Linear(hidden_state, input_size).to(torch.float64).to(device)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ff(F.relu(out))
        return out
