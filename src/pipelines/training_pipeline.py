from src.model.esn import ESN
from src.model.lstm import LSTM
from src.trainer.ridge_regression_trainer import RidgeRegressionTrainer
from src.trainer.bptt_trainer import BPTTTrainer
from src.dataset import SifimDataset
from src.trainer.model_selection import model_selection, retraining
from src.utility import fix_seed, select_device, gridsearch_generator
from tqdm import tqdm


def training_pipeline(do_model_selection=True):
    fix_seed()
    device = select_device()

    # dataset
    noise = 0.005
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True, noise=noise)
    ts_dataset = SifimDataset(start=0.8, end=1, test=True, noise=noise)

    configs = [
        ('ESN', dict(
            hyperparams_list=gridsearch_generator(
                model_reservoir_size=[100, 200],
                model_alpha=[0.5],
                model_input_ratio=[0.7, 0.9],
                model_spectral_radius=[1.2, 0.9],
                model_input_sparsity=[0.5],
                model_reservoir_sparsity=[0.9],
                model_regularization=[0.001, 0.1, 0.01],
                model_n_layers=[2, 1],
                model_washout=[100],
                model_threshold_perc=[0.8, 1, 1.2],
                model_window=[20, 50],
                model_seed=[0],
                model_requires_grad=[False],
            ),
            model_constructor=ESN,
            trainer_constructor=RidgeRegressionTrainer,
        ),),
        ('LSTM', dict(
            hyperparams_list=gridsearch_generator(
                model_hidden_state=[100, 200],
                model_n_layers=[2, 1],
                model_dropout=[0, 0.1, 0.2],
                model_threshold_perc=[0.8, 1, 1.2],
                model_window=[20, 50],
                trainer_epochs=[50],
                trainer_lr=[1e-02, 1e-03],
                trainer_momentum=[0, 0.9],
                trainer_weight_decay=[0, 0.001],
            ),
            model_constructor=LSTM,
            trainer_constructor=BPTTTrainer,
        ),),
    ]

    """configs.append(
        ('ESN_BPTT', dict(
            hyperparams_list=gridsearch_generator(
                model_reservoir_size=[200],
                model_alpha=[0.5],
                model_input_ratio=[0.7],
                model_spectral_radius=[0.9],
                model_input_sparsity=[0.5],
                model_reservoir_sparsity=[0.9],
                model_regularization=[0.001],
                model_n_layers=[1],
                model_washout=[100],
                model_threshold_perc=[1],
                model_window=[20],
                model_seed=[0],
                model_requires_grad=[True],
                trainer_epochs=[50],
                trainer_lr=[1e-02, 1e-03, 1e-04]
            ),
            model_constructor=ESN,
            trainer_constructor=BPTTTrainer,
        ),),
    )"""

    for name, config in configs:
        if do_model_selection:
            model_selection(
                **config,
                tr_dataset=tr_dataset,
                vl_dataset=vl_dataset,
                batch_size=16,
                shuffle=True,
                hyperparams_path=f'hyperparams/{name}_hyperparams.json',
                tqdm=tqdm,
            )
        retraining(
            model_constructor=config['model_constructor'],
            trainer_constructor=config['trainer_constructor'],
            tr_dataset=tr_dataset,
            ts_dataset=ts_dataset,
            batch_size=16,
            shuffle=True,
            hyperparams_path=f'hyperparams/{name}_hyperparams.json',
            model_path=f'models/{name}.torch',
            history_path='history/history.json',
            title=name,
        )
