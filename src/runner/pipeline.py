from src.dataset import SifimDataset
from src.utility import fix_seed, select_device


def pipeline():
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(start=0.0, end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8, test=True)
    ts_dataset = SifimDataset(start=0.8, end=1.0, test=True)

    # train concept drift detector

    # train concept anomaly detector

    # test phase

    # plot results


