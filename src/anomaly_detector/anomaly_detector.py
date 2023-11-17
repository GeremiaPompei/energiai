import torch

from src.anomaly_detector.lstm import LSTM
from src.anomaly_detector.lstm_trainer import LSTMTrainer
from src.dataset import SifimDataset
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

    def predict(self, ts_dataset: SifimDataset):
        self.model.eval()
        x, y = ts_dataset.x[:, 1:], ts_dataset.x[:, :-1]
        return self.model(x, y)
