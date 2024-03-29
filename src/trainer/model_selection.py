import json
import os.path

import torch
from src.utility import log

from src.utility.fix_seed import fix_seed
from src.utility.select_device import select_device


def __read_hyperparams_file__(filename, cache=None, criterion_metric=max):
    with open(filename, 'r') as fn:
        data = json.load(fn)
        best_hyperparams = data['best_hyperparams']
        best_loss = data['best_loss']
        for c in data['cache']:
            hp = json.dumps(c['hyperparams'])
            if cache is not None and len(cache) > 0 and hp not in cache:
                cache[hp] = c['loss']
    if cache is None:
        return best_hyperparams, best_loss
    else:
        for hp, loss in cache.items():
            if best_loss is None or criterion_metric(best_loss, loss) == loss:
                best_loss = loss
                best_hyperparams = json.loads(hp)
        return best_hyperparams, best_loss, cache


def model_selection(
        hyperparams_list,
        model_constructor,
        trainer_constructor,
        tr_dataset,
        vl_dataset,
        batch_size=32,
        shuffle=True,
        hyperparams_path='hyperparams/hyperparams.json',
        tqdm=None,
        metric='accuracy',
        criterion_metric=max,
):
    fix_seed()
    device = select_device()

    best_hyperparams, best_loss, cache = None, None, {}
    if os.path.exists(hyperparams_path):
        best_hyperparams, best_loss, cache = __read_hyperparams_file__(
            hyperparams_path, cache, criterion_metric)

    if tqdm is None:
        def tqdm(x): return x
    for i, hyperparams in enumerate(tqdm(hyperparams_list)):
        if json.dumps(hyperparams) in cache:
            vl_loss = cache[json.dumps(hyperparams)]
        else:
            log.info(
                f'Start iteration {i + 1}/{len(hyperparams_list)} => hyperparams: {hyperparams}')
            model_hyperparams = {
                k.replace('model_', ''): v for k, v in hyperparams.items() if 'model_' in k}
            trainer_hyperparams = {k.replace(
                'trainer_', ''): v for k, v in hyperparams.items() if 'trainer_' in k}

            tr_dataloader = torch.utils.data.DataLoader(
                tr_dataset, batch_size=batch_size, shuffle=shuffle)
            vl_dataloader = torch.utils.data.DataLoader(
                vl_dataset, batch_size=batch_size, shuffle=shuffle)

            # model
            model = model_constructor(
                tr_dataset.x.shape[-1], **model_hyperparams, device=device)

            # trainer
            trainer = trainer_constructor(
                model, tr_dataloader, vl_dataloader, device=device)
            vl_loss = trainer(**trainer_hyperparams)[metric]

        cache[json.dumps(hyperparams)] = vl_loss
        if os.path.exists(hyperparams_path):
            best_hyperparams, best_loss, cache = __read_hyperparams_file__(
                hyperparams_path, cache, criterion_metric)
        if best_loss is None or criterion_metric(best_loss, vl_loss) == vl_loss:
            best_hyperparams = hyperparams
            best_loss = vl_loss
            log.info(
                f'Best loss: {best_loss}, best hyperparams: {best_hyperparams}')
        with open(hyperparams_path, 'w') as fn:
            json.dump(dict(
                best_hyperparams=best_hyperparams,
                best_loss=best_loss,
                cache=[
                    dict(hyperparams=json.loads(hyperparams), loss=loss)
                    for hyperparams, loss in cache.items()
                ],
            ), fn)

    log.info(
        f'Final best loss: {best_loss}, best hyperparams: {best_hyperparams}')


def retraining(
        model_constructor,
        trainer_constructor,
        tr_dataset,
        ts_dataset,
        batch_size=32,
        shuffle=True,
        model_path='model/model.torch',
        history_path='history/',
        hyperparams_path='hyperparams/hyperparams.json',
        best_hyperparams=None,
        title=None,
):
    fix_seed()
    device = select_device()
    if best_hyperparams is None and os.path.exists(hyperparams_path):
        best_hyperparams, _ = __read_hyperparams_file__(hyperparams_path)
    model_hyperparams = {k.replace('model_', ''): v for k, v in best_hyperparams.items() if 'model_' in k}
    trainer_hyperparams = {k.replace('trainer_', ''): v for k, v in best_hyperparams.items() if 'trainer_' in k}
    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=shuffle)
    ts_dataloader = torch.utils.data.DataLoader(
        ts_dataset, batch_size=batch_size, shuffle=shuffle)
    # model
    model = model_constructor(
        tr_dataset.x.shape[-1], **model_hyperparams, device=device)
    # trainer
    trainer = trainer_constructor(
        model, tr_dataloader, ts_dataloader, device=device)
    if model_path is None or not os.path.exists(model_path):
        history = trainer(**trainer_hyperparams, save_emissions=history_path)
        total_history = {}
        history_fn = f'{history_path}history.json'
        if history_path is not None:
            if os.path.exists(history_fn):
                total_history = json.load(open(history_fn))
            total_history[title if title is not None else model.__class__.__name__] = history
            with open(history_fn, 'w') as fp:
                json.dump(total_history, fp)

        if model_path is not None:
            torch.save(model, model_path)
    else:
        model = torch.load(model_path)
    return model
