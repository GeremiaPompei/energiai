from src.concept_drift_detector.concept_drift_detector import ConceptDriftDetector
from src.dataset import SifimDataset
from src.utility import fix_seed, select_device, gridsearch_generator


def pipeline():
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True)
    ts_dataset = SifimDataset(start=0.8, end=1.0, test=True)

    # train concept drift detector
    cdd = ConceptDriftDetector(
        hyperparams_list=gridsearch_generator(
            window=[10],
            hidden_dim=[200],
            latent_dim=[10],
            n_layers=[0],
            bias_perc_thresh=[-0.8]
        ),
        tr_dataset=tr_dataset,
        vl_dataset=vl_dataset,
        epochs=20,
        batch_size=32,
        shuffle=True,
        hyperparams_path='hyperparams/hyperparams.json',
        tqdm=None,
        retrain=True,
    )

    # train concept anomaly_detector detector

    # test phase
    cdd.predict(ts_dataset)

    # plot results
