import torch

from src.dataset import SifimDataset
from src.model import VAE
from src.preprocessing import preprocessing_pipeline
from src.trainer.vae_trainer import VAETrainer
from src.utility.fix_seed import fix_seed
from src.utility.select_device import select_device


def training_pipeline(epochs=100, batch_size=64, lr=1e-3, hidden_dim=400, latent_dim=200, shuffle=True):
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
    vl_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=batch_size, shuffle=shuffle)

    # model
    model = VAE(tr_dataset.dataset.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)

    # trainer
    trainer = VAETrainer(model, tr_dataloader, vl_dataloader, device=device)
    trainer(epochs=epochs, lr=lr)
