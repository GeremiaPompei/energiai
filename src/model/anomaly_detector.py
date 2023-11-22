import torch.nn
import torch


class AnomalyDetector(torch.nn.Module):

    def __init__(self, *args, window: int = 10, threshold_perc: float = 1.2, **hyperparams: dict):
        super(AnomalyDetector, self).__init__()
        self.window = window
        self.sigma = 0
        self.tr_data = 0
        self.threshold_perc = threshold_perc

    def _loss_(self, y, p):
        return (y - p) ** 2

    def reset(self):
        self.sigma = 0
        self.tr_data = 0

    @torch.no_grad()
    def register_std(self, x, y):
        out = self(x)
        e = self._loss_(y, out).reshape(-1, y.shape[-1]).std(0)
        tot = self.tr_data + 1
        self.sigma = (self.sigma * self.tr_data + e) / tot
        self.tr_data = tot

    @torch.no_grad()
    def predict(self, x, y):
        out = self(x)
        e = self._loss_(y, out)
        e_std = e.unfold(1, self.window, 1).std(-1)
        res = torch.logical_or(-self.threshold_perc * self.sigma > e_std, e_std > self.threshold_perc * self.sigma).to(
            torch.float64)
        return res, out.detach(), e_std.detach()
