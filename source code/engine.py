"""Training and evaluation loops."""

import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, covid_loss, source_loss, scaler, device, gamma):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels, sources in tqdm(loader, desc='train', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        sources = sources.to(device)

        optimizer.zero_grad()
        with autocast():
            covid_out, source_out = model(images)
            l_covid = covid_loss(covid_out, labels)
            l_source = source_loss(source_out, sources)
            loss = l_covid + gamma * l_source

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(covid_out).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = (np.array(all_preds) >= 0.5).astype(int)
    f1 = f1_score(np.array(all_labels), preds)
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, covid_loss, source_loss, device, gamma):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels, sources in tqdm(loader, desc='val', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        sources = sources.to(device)

        with autocast():
            covid_out, source_out = model(images)
            l_covid = covid_loss(covid_out, labels)
            l_source = source_loss(source_out, sources)
            loss = l_covid + gamma * l_source

        total_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(covid_out).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return avg_loss, f1, auc, probs, labels


def compute_per_source_metrics(val_data, probs, labels):
    """Compute per-source F1 and the competition final score.

    Final score = (1/|D|) * sum_i (F1_covid_i + F1_noncovid_i) / 2
    """
    preds = (probs >= 0.5).astype(int)
    source_ids = sorted(set(s['source'] for s in val_data))

    results = {}
    total_score = 0.0

    for sid in source_ids:
        idx = [i for i, s in enumerate(val_data) if s['source'] == sid]
        y = labels[idx].astype(int)
        p = preds[idx]

        f1_covid = f1_score(y, p, pos_label=1) if (y == 1).sum() > 0 else 0.0
        f1_noncovid = f1_score(y, p, pos_label=0) if (y == 0).sum() > 0 else 0.0
        avg = (f1_covid + f1_noncovid) / 2

        results[sid] = {
            'f1_covid': f1_covid,
            'f1_noncovid': f1_noncovid,
            'avg': avg,
            'n_scans': len(idx),
        }
        total_score += avg

    final_score = total_score / len(source_ids)
    return results, final_score
