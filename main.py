from src.pipelines.preprocessing_pipeline import preprocessing_pipeline
from src.pipelines.training_pipeline import training_pipeline


def main():
    preprocessing_pipeline()
    training_pipeline()


if __name__ == '__main__':
    main()
