import time

import torch
from src.utility import log
from src.utility.metrics import compute_scores
from codecarbon import EmissionsTracker


class Trainer:
    def __init__(self, model, tr_loader, ts_loader, device='cpu'):
        self.model = model
        self.tr_loader = tr_loader
        self.ts_loader = ts_loader
        self.device = device
        self.criterion = torch.nn.MSELoss()

    def train_model(self, *args, **kwargs):
        pass

    def __construct_emissions_tracker__(self, *args, **kwargs) -> EmissionsTracker:
        name = f'{self.model.__class__.__name__}_{self.__class__.__name__}_{kwargs["type"]}'
        return EmissionsTracker(
            log_level="critical",
            save_to_file='save_emissions' in kwargs,
            output_file=f'{kwargs["save_emissions"]}emissions.csv' if 'save_emissions' in kwargs else '',
            project_name=name,
        )

    def test(self, *args, **kwargs):
        for batch_idx, (data, _) in enumerate(self.tr_loader):
            data = data.to(self.device)
            x, y = data[:, 1:], data[:, :-1]
            self.model.compute_batch_std(x, y)
        self.model.compute_std()

        outputs = []
        emissions_tracker = self.__construct_emissions_tracker__(*args, type='test', **kwargs)
        emissions_tracker.start()
        start = time.time()
        for batch_idx, (data, labels) in enumerate(self.ts_loader):
            data = data.to(self.device)
            x, y = data[:, 1:], data[:, :-1]
            p, o, _ = self.model.predict(x, y)
            outputs.append((y, o, labels, p))
        ts_time = time.time() - start
        ts_emissions = emissions_tracker.stop()

        ts_loss, scores = 0, None
        for y, o, labels, p in outputs:
            ts_loss += self.criterion(y, o).item()
            labels = labels[:, -p.shape[1]:]
            curr_scores = compute_scores(labels, p)
            if scores is not None:
                for k, v in curr_scores.items():
                    scores[k] += v
            else:
                scores = curr_scores
        ts_loss = ts_loss / len(self.ts_loader)
        scores = {k: s / len(self.ts_loader) for k, s in scores.items()}
        return ts_loss, ts_time, ts_emissions, scores

    def __call__(self, *args, **kwargs):
        self.model.train()

        emissions_tracker = self.__construct_emissions_tracker__(*args, type='train', **kwargs)
        emissions_tracker.start()
        start = time.time()
        tr_loss = self.train_model(*args, **kwargs)
        tr_time = time.time() - start
        tr_emissions = emissions_tracker.stop()

        ts_loss, ts_time, ts_emissions, scores = self.test(*args, **kwargs)

        log.info(
            f'{self.model.__class__.__name__} results => training loss: {tr_loss}, test loss: {ts_loss}, '
            f'tr_time: {tr_time}, tr_emissions: {tr_emissions}, ts_time: {ts_time}, ts_emissions: {ts_emissions}, '
            f'scores: {scores}')
        return dict(tr_loss=tr_loss, ts_loss=ts_loss, tr_time=tr_time, tr_emissions=tr_emissions, ts_time=ts_time,
                    ts_emissions=ts_emissions, **scores)
