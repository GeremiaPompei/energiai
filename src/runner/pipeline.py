import torch

from src.anomaly_detector.anomaly_detector import AnomalyDetector
from src.concept_drift_detector.concept_drift_detector import ConceptDriftDetector
from src.dataset import SifimDataset
from src.plot.timeseries_analysis import plot_with_thresholds
from src.utility import fix_seed, select_device, gridsearch_generator


def pipeline():
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True)
    ts_dataset = SifimDataset(start=0.8, end=1.0, test=True, noise=0.01)

    # train concept anomaly_detector detector
    ad = AnomalyDetector(
        hyperparams_list=gridsearch_generator(
            hidden_state=[100],  # , 200, 300],
            ff_size=[300],  # , 500, 1000],
            window=[10],
            n_layers=[1],  # , 2, 3],
            bidirectional=[False],  # , True],
        ),
        tr_dataset=tr_dataset,
        vl_dataset=vl_dataset,
        epochs=20,
        batch_size=32,
        shuffle=True,
        hyperparams_path='hyperparams/ad_hyperparams.json',
        model_path='models/lstm.torch',
        tqdm=None,
        retrain=True,
    )

    # test phase
    ad_labels, ad_predictions, ad_std = ad.predict(ts_dataset)

    """i = 0
    window = 75
    for f in range(1):
        y, p = ts_dataset.x[i, -window:, f], ad_predictions[i, -window:, f]
        plot_with_thresholds(y, p)"""

    # plot
    window = 75
    for i in range(5):
        std = ad_std[i, -window:]
        m2s = torch.ones_like(std) * ad.model.sigma.item() * (-2)
        p2s = torch.ones_like(std) * ad.model.sigma.item() * (+2)
        plot_with_thresholds('Standard deviation', [std, m2s, p2s])  # standard deviation

        for f in range(1):
            y, p = ts_dataset.x[i, -window:, f], ad_predictions[i, -window:, f]
            plot_with_thresholds('Timeseries prediction', [y, p]) # timeseries and prediction of lstm

        plot_with_thresholds('Labels', [ts_dataset.y[i, -ad_labels.shape[1]:], ad_labels[i]]) # labels
