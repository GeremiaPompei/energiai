import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_state,
            ff_size,
            n_layers: int = 0,
            bidirectional: bool = False,
            device='cpu',
    ):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_state, num_layers=n_layers, bidirectional=bidirectional).to(device)
        self.ff1 = torch.nn.Linear(hidden_state, ff_size).to(device)
        self.ff2 = torch.nn.Linear(ff_size, output_size).to(device)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = F.relu(out)
        out = F.relu(self.ff1(out))
        out = self.ff2(out)
        return out
