from src.model.esn import ESN
from src.model.lstm import LSTM
from src.trainer.esn_trainer import ESNTrainer
from src.trainer.lstm_trainer import LSTMTrainer
from src.dataset import SifimDataset
from src.trainer.model_selection import model_selection
from src.utility import fix_seed, select_device, gridsearch_generator


def training_pipeline():
    fix_seed()
    device = select_device()

    # dataset
    noise = 0.1
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True, noise=noise)
    ts_dataset = SifimDataset(start=0.8, end=1, test=True, noise=noise)

    configs = [
        ('ESN', dict(
            hyperparams_list=gridsearch_generator(
                model_reservoir_size=[100],
                model_alpha=[0.5, 0.1],
                model_input_ratio=[0.7, 0.9],
                model_spectral_radius=[1.2, 0.9],
                model_input_sparsity=[0.5],
                model_reservoir_sparsity=[0.9],
                model_regularization=[0.001, 0.1, 0.01],
                model_n_layers=[2, 1],
                model_washout=[100],
                model_seed=[0]
            ),
            model_constructor=ESN,
            trainer_constructor=ESNTrainer,
        ),),
        ('LSTM', dict(
            hyperparams_list=gridsearch_generator(
                model_hidden_state=[100, 200, 300],
                model_ff_size=[300, 500, 1000],
                model_n_layers=[1, 2, 3],
                trainer_epochs=[20],
            ),
            model_constructor=LSTM,
            trainer_constructor=LSTMTrainer,
        ),),
    ]

    for name, config in configs:
        model_selection(
            **config,
            tr_dataset=tr_dataset,
            vl_dataset=vl_dataset,
            ts_dataset=ts_dataset,
            batch_size=16,
            shuffle=True,
            hyperparams_path=f'hyperparams/{name}_hyperparams.json',
            model_path=None,  # f'models/{name}.torch',
            tqdm=None,
            retrain=True,
        )
