import json
from datetime import datetime

import torch
from tqdm import tqdm

from src.dataset import SifimDataset
from src.model import VAE
from src.trainer.vae_trainer import VAETrainer
from src.utility.fix_seed import fix_seed
from src.utility.select_device import select_device


def model_selection_pipeline(hyperparams_list, epochs=100, batch_size=64, shuffle=True):
    filename = f'hyperparams/{datetime.now()}.json'
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(end=0.6)
    vl_dataset = SifimDataset(start=0.6, end=0.8)

    best_hyperparams, best_loss = None, None
    pbar = tqdm(hyperparams_list, desc='model selection')
    for hyperparams in pbar:
        pbar.write(f'hyperparams: {hyperparams}')
        model_hyperparams = {k.replace('model_', ''): v for k, v in hyperparams.items() if 'model_' in k}
        trainer_hyperparams = {k.replace('trainer_', ''): v for k, v in hyperparams.items() if 'trainer_' in k}

        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
        vl_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=batch_size, shuffle=shuffle)

        # model
        model = VAE(tr_dataset.dataset.shape[1], **model_hyperparams, device=device)

        # trainer
        trainer = VAETrainer(model, tr_dataloader, vl_dataloader, device=device)

        tr_loss, vl_loss = trainer(epochs=epochs, **trainer_hyperparams)

        if best_loss is None or best_loss > vl_loss:
            best_hyperparams = hyperparams
            best_loss = vl_loss
            with open(filename, 'w') as fn:
                json.dump(dict(best_hyperparams=best_hyperparams, best_loss=best_loss), fn)
