import torch

from src.anomaly_detector.lstm import LSTM
from src.anomaly_detector.lstm_trainer import LSTMTrainer
from src.detector.detector import Detector
from src.detector.model_selection import model_selection


class AnomalyDetector(Detector):

    def _train(self, *args, **kwargs):
        self.model = model_selection(
            *args,
            model_constructor=LSTM,
            trainer_constructor=LSTMTrainer,
            **kwargs,
        )

    def predict(self, ts_dataset, batch_size=32, shuffle=True):
        ts_dataloader = torch.utils.data.DataLoader(ts_dataset, batch_size=batch_size, shuffle=shuffle)
        self.model.eval()
        for batch_idx, (x, y) in enumerate(ts_dataloader):
            m, v = self.model.encode(x.to(self.device))
            z = self.model.reparameterization(m, v).to(torch.float64).abs()
            I = torch.eye(z.shape[-2]).to(self.device).to(torch.float64)
            dist = (z.transpose(-1, -2) @ I @ z).sqrt()
            print(dist.mean())
