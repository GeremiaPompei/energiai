import torch
import torch.nn.functional as F

from src.model.anomaly_detector import AnomalyDetector


class LSTM(AnomalyDetector):
    def __init__(
            self,
            input_size,
            hidden_state: int = 100,
            ff_size: int = 300,
            n_layers: int = 1,
            bidirectional: bool = False,
            device='cpu',
            **hyperparams: dict,
    ):
        super(LSTM, self).__init__(**hyperparams)
        self.lstm = torch.nn.LSTM(input_size, hidden_state, num_layers=n_layers, bidirectional=bidirectional,
                                  batch_first=True).to(torch.float64).to(device)
        self.ff1 = torch.nn.Linear(hidden_state, ff_size).to(torch.float64).to(device)
        self.ff2 = torch.nn.Linear(ff_size, input_size).to(torch.float64).to(device)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = F.relu(out)
        out = F.relu(self.ff1(out))
        out = self.ff2(out)
        return out
