import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_state: int = 100,
            ff_size: int = 300,
            window: int = 10,
            n_layers: int = 1,
            bidirectional: bool = False,
            device='cpu',
    ):
        super(LSTM, self).__init__()
        self.window = window
        self.lstm = torch.nn.LSTM(input_size, hidden_state, num_layers=n_layers, bidirectional=bidirectional,
                                  batch_first=True).to(torch.float64).to(device)
        self.ff1 = torch.nn.Linear(hidden_state, ff_size).to(torch.float64).to(device)
        self.ff2 = torch.nn.Linear(ff_size, input_size).to(torch.float64).to(device)
        self.sigma = 0
        self.tr_data = 0

    def reset(self):
        self.sigma = 0
        self.tr_data = 0

    def __loss__(self, y, p):
        return ((y - p) ** 2).mean(-1)

    def forward(self, x, y):
        out, _ = self.lstm(x)
        out = F.relu(out)
        out = F.relu(self.ff1(out))
        out = self.ff2(out)
        e = self.__loss__(y, out)
        if self.training:
            tot = self.tr_data + 1
            self.sigma = (self.sigma * self.tr_data + e.std()) / tot
            self.tr_data = tot
            return out
        else:
            e_std = e.unfold(1, self.window, 1).std(-1)
            res = torch.logical_or(-2 * self.sigma > e_std, e_std > 2 * self.sigma).to(torch.float64)
            return res, out.detach(), e_std.detach()
