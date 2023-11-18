import torch.nn


class AnomalyDetector(torch.nn.Module):

    def __init__(self, window: int = 10):
        super(AnomalyDetector, self).__init__()
        self.window = window
        self.sigma = 0
        self.tr_data = 0

    def _loss_(self, y, p):
        return ((y - p) ** 2).mean(-1)

    def reset(self):
        self.sigma = 0
        self.tr_data = 0

    def register_std(self, x, y):
        out = self(x)
        e = self._loss_(y, out)
        tot = self.tr_data + 1
        self.sigma = (self.sigma * self.tr_data + e.std()) / tot
        self.tr_data = tot

    def predict(self, x, y):
        out = self(x)
        e = self._loss_(y, out)
        e_std = e.unfold(1, self.window, 1).std(-1)
        res = torch.logical_or(-2 * self.sigma > e_std, e_std > 2 * self.sigma).to(torch.float64)
        return res, out.detach(), e_std.detach()
