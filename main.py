import random
from tqdm import tqdm

from src.preprocessing import preprocessing_pipeline
from src.runner import training_pipeline
from src.runner.model_selection_pipeline import model_selection_pipeline
from src.utility import fix_seed
from src.utility.hyperparams_generator import gridsearch_generator, randomsearch_generator


def main():
    fix_seed(seed=0)

    # preprocessing
    preprocessing_pipeline()

    # training
    # training_pipeline()

    # gridsearch
    hyperparams_list = gridsearch_generator(
        dict(
            trainer_lr=[0.001, 0.0001],
            model_hidden_dim=[300, 200, 100, 50],
            model_latent_dim=[100, 75, 50],
            model_n_layers=[0, 1, 2],
        )
    )
    # random search
    hyperparams_list += [
        dict(
            trainer_lr=random.uniform(0, 0.001),
            model_hidden_dim=random.randint(20, 300),
            model_latent_dim=random.randint(20, 200),
            model_n_layers=random.randint(0, 4),
        ) for _ in range(100)
    ]
    model_selection_pipeline(
        hyperparams_list,
        epochs=50,
        batch_size=128,
        shuffle=False,
        tqdm=tqdm,
        hyperparams_path='hyperparams/hyperparams_linear_bn_ns.json',
    )


if __name__ == '__main__':
    main()
