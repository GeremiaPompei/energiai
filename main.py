import random

from src.preprocessing import preprocessing_pipeline
from src.runner import training_pipeline
from src.runner.model_selection_pipeline import model_selection_pipeline
from src.utility.hyperparams_generator import gridsearch_generator, randomsearch_generator


def main():
    # preprocessing
    # preprocessing_pipeline()

    # training
    # training_pipeline()

    gridsearch = True
    if gridsearch:
        hyperparams_list = gridsearch_generator(
            dict(
                trainer_lr=[0.001, 0.1],
                model_hidden_dim=[400],
                model_latent_dim=[200],
            )
        )
    else:
        hyperparams_list = [
            dict(
                trainer_lr=random.uniform(0, 1),
                model_hidden_dim=random.randint(100, 600),
                model_latent_dim=random.randint(100, 600),
            ) for _ in range(100)
        ]
    model_selection_pipeline(hyperparams_list)


if __name__ == '__main__':
    main()
