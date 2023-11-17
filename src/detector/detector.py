from abc import ABC, abstractmethod


class Detector(ABC):

    def __init__(self, *args, device: str = 'cpu', **kwargs):
        self.device = device
        self._train(*args, **kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, ts_dataset):
        pass
