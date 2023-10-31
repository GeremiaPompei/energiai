import json
import os.path

import torch
from src.utility import log

from src.dataset import SifimDataset
from src.model import VAE
from src.trainer.vae_trainer import VAETrainer
from src.utility.fix_seed import fix_seed
from src.utility.select_device import select_device


def model_selection_pipeline(hyperparams_list, epochs=20, batch_size=32, shuffle=True, dataset_dir='dataset/cleaned/',
                             hyperparams_dir='hyperparams', tqdm=None):
    if not os.path.exists(hyperparams_dir):
        os.mkdir(hyperparams_dir)
    filename = f'{hyperparams_dir}/hyperparams.json'
    fix_seed()
    device = select_device()

    # dataset
    tr_dataset = SifimDataset(dir=dataset_dir, end=0.8)
    vl_dataset = SifimDataset(dir=dataset_dir, start=0.8)

    best_hyperparams, best_loss = None, None
    cache = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fn:
            data = json.load(fn)
            best_hyperparams = data['best_hyperparams']
            best_loss = data['best_loss']
            cache = data['cache']
            cache = {json.dumps(c['hyperparams']): c['loss'] for c in cache}

    if tqdm is None:
        tqdm = lambda x: x
    for i, hyperparams in enumerate(tqdm(hyperparams_list)):
        log.info(f'Start iteration {i + 1}/{len(hyperparams_list)} => hyperparams: {hyperparams}')
        model_hyperparams = {k.replace('model_', ''): v for k, v in hyperparams.items() if 'model_' in k}
        trainer_hyperparams = {k.replace('trainer_', ''): v for k, v in hyperparams.items() if 'trainer_' in k}

        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
        vl_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=batch_size, shuffle=shuffle)

        # model
        model = VAE(tr_dataset.dataset.shape[-1], tr_dataset.dataset.shape[1], 1, **model_hyperparams, device=device)

        # trainer
        trainer = VAETrainer(model, tr_dataloader, vl_dataloader, device=device)

        if json.dumps(hyperparams) in cache:
            vl_loss = cache[json.dumps(hyperparams)]
        else:
            _, vl_loss = trainer(epochs=epochs, **trainer_hyperparams)

        if best_loss is None or best_loss > vl_loss:
            best_hyperparams = hyperparams
            best_loss = vl_loss
            formatted_cache = [dict(hyperparams=json.loads(hyperparams), loss=loss) for hyperparams, loss in
                               cache.items()]
            with open(filename, 'w') as fn:
                json.dump(dict(
                    best_hyperparams=best_hyperparams,
                    best_loss=best_loss,
                    cache=formatted_cache,
                ), fn)
                log.info(f'Best loss: {best_loss}, best hyperparams: {best_hyperparams}')

    log.info(f'Final best loss: {best_loss}, best hyperparams: {best_hyperparams}')
