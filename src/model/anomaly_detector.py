import torch.nn
import torch


class AnomalyDetector(torch.nn.Module):

    def __init__(self, *args, window: int = 10, threshold_perc: float = 1.2, **hyperparams: dict):
        super(AnomalyDetector, self).__init__()
        self.window = window
        self.sigma = 0
        self.batch_stds = []
        self.threshold_perc = threshold_perc

    def _loss_(self, y, p):
        return (y - p) ** 2

    def reset(self):
        self.sigma = 0
        self.batch_stds = []

    @torch.no_grad()
    def compute_batch_std(self, x, y):
        out = self(x)
        e = self._loss_(y, out).reshape(-1, y.shape[-1])
        self.batch_stds.append(e)

    @torch.no_grad()
    def compute_std(self):
        self.sigma = torch.cat(self.batch_stds, dim=0).std(0)

    @torch.no_grad()
    def predict(self, x, y):
        out = self(x)
        e = self._loss_(y, out)
        e_std = e.unfold(1, self.window, 1).std(-1)
        res = torch.logical_or(-self.threshold_perc * self.sigma > e_std, e_std > self.threshold_perc * self.sigma).to(
            torch.float64)
        return res, out.detach(), e_std.detach()
