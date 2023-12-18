from src.trainer.trainer import Trainer


class RidgeRegressionTrainer(Trainer):

    def train_model(self, *args, **kwargs):
        tr_loss = 0
        self.model.train()
        for batch_idx, (data, _) in enumerate(self.tr_loader):
            data = data.to(self.device)
            A, B = self.model.compute_ab(data[:, 1:], data[:, :-1])
            if self.model.A is not None:
                A += self.model.A
                B += self.model.B
            self.model.fit_from_ab(A, B)
            x, y = data[:, 1:], data[:, :-1]
            p = self.model(x)
            tr_loss += self.criterion(y, p).item()
        tr_loss = tr_loss / len(self.tr_loader)
        return tr_loss
