import json
import os.path

import torch
from src.utility import log

from src.utility.fix_seed import fix_seed
from src.utility.select_device import select_device


def __read_hyperparams_file__(filename, cache):
    with open(filename, 'r') as fn:
        data = json.load(fn)
        best_hyperparams = data['best_hyperparams']
        best_loss = data['best_loss']
        for c in data['cache']:
            hp = json.dumps(c['hyperparams'])
            if hp not in cache:
                cache[hp] = c['loss']
    for hp, loss in cache.items():
        if best_loss is None or best_loss > loss:
            best_loss = loss
            best_hyperparams = json.loads(hp)
    return best_hyperparams, best_loss, cache


def model_selection(
        hyperparams_list,
        model_constructor,
        trainer_constructor,
        tr_dataset,
        vl_dataset,
        epochs=20,
        batch_size=32,
        shuffle=True,
        hyperparams_path='hyperparams/hyperparams.json',
        model_path='models/model.torch',
        tqdm=None,
        retrain=True,
):
    filename = hyperparams_path
    fix_seed()
    device = select_device()

    best_hyperparams, best_loss, cache = None, None, {}
    if os.path.exists(filename):
        best_hyperparams, best_loss, cache = __read_hyperparams_file__(filename, cache)

    if tqdm is None:
        tqdm = lambda x: x
    for i, hyperparams in enumerate(tqdm(hyperparams_list)):
        log.info(f'Start iteration {i + 1}/{len(hyperparams_list)} => hyperparams: {hyperparams}')
        model_hyperparams = {k.replace('model_', ''): v for k, v in hyperparams.items() if 'model_' in k}
        trainer_hyperparams = {k.replace('trainer_', ''): v for k, v in hyperparams.items() if 'trainer_' in k}

        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
        vl_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=batch_size, shuffle=shuffle)

        # model
        model = model_constructor(tr_dataset.x.shape[-1], **model_hyperparams, device=device)

        # trainer
        trainer = trainer_constructor(model, tr_dataloader, vl_dataloader, device=device)

        if json.dumps(hyperparams) in cache:
            vl_loss = cache[json.dumps(hyperparams)]
        else:
            _, vl_loss = trainer(epochs=epochs, **trainer_hyperparams)

        cache[json.dumps(hyperparams)] = vl_loss
        if os.path.exists(filename):
            best_hyperparams, best_loss, cache = __read_hyperparams_file__(filename, cache)
        if best_loss is None or best_loss > vl_loss:
            best_hyperparams = hyperparams
            best_loss = vl_loss
            log.info(f'Best loss: {best_loss}, best hyperparams: {best_hyperparams}')
        with open(filename, 'w') as fn:
            json.dump(dict(
                best_hyperparams=best_hyperparams,
                best_loss=best_loss,
                cache=[
                    dict(hyperparams=json.loads(hyperparams), loss=loss)
                    for hyperparams, loss in cache.items()
                ],
            ), fn)

    log.info(f'Final best loss: {best_loss}, best hyperparams: {best_hyperparams}')

    if retrain:
        model_hyperparams = {k.replace('model_', ''): v for k, v in best_hyperparams.items() if 'model_' in k}
        trainer_hyperparams = {k.replace('trainer_', ''): v for k, v in best_hyperparams.items() if 'trainer_' in k}
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
        vl_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=batch_size, shuffle=shuffle)
        # model
        model = model_constructor(tr_dataset.x.shape[-1], **model_hyperparams, device=device)
        # trainer
        trainer = trainer_constructor(model, tr_dataloader, vl_dataloader, device=device)
        if not os.path.exists(model_path):
            _, vl_loss = trainer(epochs=epochs, **trainer_hyperparams)
            torch.save(model, model_path)
        else:
            model = torch.load(model_path)
        return model
