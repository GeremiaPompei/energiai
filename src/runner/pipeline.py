import torch

from src.model.esn import ESN
from src.model.lstm import LSTM
from src.plot.timeseries_analysis import plot_with_thresholds
from src.trainer.esn_trainer import ESNTrainer
from src.trainer.lstm_trainer import LSTMTrainer
from src.dataset import SifimDataset
from src.model_selection.model_selection import model_selection
from src.utility import fix_seed, select_device, gridsearch_generator


def pipeline():
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True, noise=0.1)
    ts_dataset = SifimDataset(start=0.8, end=1.0, test=True, noise=0.1)
    x_ts, y_ts = ts_dataset.x[:, :-1], ts_dataset.x[:, 1:]

    configs = [
        ('LSTM', dict(
            hyperparams_list=gridsearch_generator(
                model_reservoir_size=[100],
                model_alpha=[0.5, 0.1],
                model_input_ratio=[0.7, 0.9],
                model_spectral_radius=[1.2, 0.9],
                model_input_sparsity=[0.5],
                model_reservoir_sparsity=[0.9],
                model_regularization=[0.001],
                model_n_layers=[2, 1],
                model_washout=[100],
                model_seed=[0]
            ),
            model_constructor=ESN,
            trainer_constructor=ESNTrainer,
        ),),
        ('ESN', dict(
            hyperparams_list=gridsearch_generator(
                model_hidden_state=[100],  # 200, 300],
                model_ff_size=[300],  # 500, 1000],
                model_window=[10],
                model_n_layers=[1],  # 2, 3],
                trainer_epochs=[20],
            ),
            model_constructor=LSTM,
            trainer_constructor=LSTMTrainer,
        ),),
    ]

    for name, config in configs:
        # train concept model model_selection
        model = model_selection(
            **config,
            tr_dataset=tr_dataset,
            vl_dataset=vl_dataset,
            batch_size=128,
            shuffle=True,
            hyperparams_path=f'hyperparams/{name}_hyperparams.json',
            model_path=None,  # 'model/lstm.torch',
            tqdm=None,
            retrain=True,
        )

        # test phase
        ad_labels, ad_predictions, ad_std = model.predict(x_ts, y_ts)

        # plot
        window = 200
        i = 0
        f = 0

        std = ad_std[i, -window:]
        m2s = torch.ones_like(std) * model.sigma.item() * (-2)
        p2s = torch.ones_like(std) * model.sigma.item() * (+2)
        plot_with_thresholds('Standard deviation', [std, m2s, p2s])  # standard deviation

        y, p = ts_dataset.x[i, -window:, f], ad_predictions[i, -window:, f]
        plot_with_thresholds('Timeseries prediction', [y, p])  # timeseries and prediction of lstm

        plot_with_thresholds('Labels', [ts_dataset.y[i, -ad_labels.shape[1]:], ad_labels[i]])  # labels
