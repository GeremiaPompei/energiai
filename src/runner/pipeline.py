from src.anomaly_detector.anomaly_detector import AnomalyDetector
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

    # train concept anomaly_detector detector
    ad = AnomalyDetector(
        hyperparams_list=gridsearch_generator(
            hidden_state=[100, 200, 300],
            ff_size=[300, 500, 1000],
            window=[10],
            n_layers=[1, 2, 3],
            bidirectional=[False, True],
        ),
        tr_dataset=tr_dataset,
        vl_dataset=vl_dataset,
        epochs=10,
        batch_size=32,
        shuffle=True,
        hyperparams_path='hyperparams/ad_hyperparams.json',
        tqdm=None,
        retrain=True,
    )

    # test phase

    # plot results
