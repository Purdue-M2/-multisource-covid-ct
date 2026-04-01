"""
Evaluate a trained checkpoint on the validation set.

Usage:
    python evaluate.py \
        --checkpoint checkpoints/best_gamma0.5.pth \
        --data_dir data/preprocessed \
        --csv_dir data/

Produces:
    - Overall metrics (accuracy, F1, AUC, sensitivity, specificity)
    - Confusion matrix
    - Per-source F1 breakdown
    - Competition score
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report,
)
import pandas as pd

from src import (
    MultiTaskEfficientNet,
    CovidCTDataset,
    get_val_transforms,
    compute_per_source_metrics,
)


SOURCE_COUNTS = [328, 330, 330, 234]
SOURCE_TOTAL = sum(SOURCE_COUNTS)
SOURCE_FREQS = [c / SOURCE_TOTAL for c in SOURCE_COUNTS]


def get_source_map(covid_csv, noncovid_csv):
    source_map = {}
    for _, row in covid_csv.iterrows():
        source_map[row['ct_scan_name']] = int(row['data_centre'])
    for _, row in noncovid_csv.iterrows():
        source_map[row['ct_scan_name']] = int(row['data_centre'])
    return source_map


def build_scanlist(base_path, source_map, prefix=''):
    scans = []
    for label_name in ['covid', 'non-covid']:
        label = 1 if label_name == 'covid' else 0
        label_path = os.path.join(base_path, label_name)
        if not os.path.exists(label_path):
            continue
        for scan_id in sorted(os.listdir(label_path)):
            scan_path = os.path.join(label_path, scan_id)
            if not os.path.isdir(scan_path):
                continue
            pngs = [f for f in os.listdir(scan_path) if f.endswith('.png')]
            if len(pngs) == 0:
                continue
            clean_name = scan_id.replace(prefix, '') if prefix else scan_id
            source = source_map.get(clean_name, source_map.get(scan_id, -1))
            if source == -1 and '_' in scan_id:
                alt = scan_id.split('_', 1)[-1]
                source = source_map.get(alt, -1)
            scans.append({
                'scan_id': scan_id,
                'label': label,
                'label_name': label_name,
                'path': scan_path,
                'source': source if source != -1 else 0,
            })
    return scans


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load source maps
    val_covid = pd.read_csv(os.path.join(args.csv_dir, 'validation_covid.csv'))
    val_noncovid = pd.read_csv(os.path.join(args.csv_dir, 'validation_non_covid.csv'))
    val_srcmap = get_source_map(val_covid, val_noncovid)

    # Build scan list
    val_data = build_scanlist(
        os.path.join(args.data_dir, 'val'), val_srcmap, prefix='val_'
    )
    print(f'Validation: {len(val_data)} scans')

    # Data loader
    val_loader = DataLoader(
        CovidCTDataset(val_data, get_val_transforms()),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
    )

    # Load model
    model = MultiTaskEfficientNet(pretrained=False, num_sources=4).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Inference
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, sources in val_loader:
            images = images.to(device)
            with autocast():
                covid_out, _ = model(images)
            all_probs.extend(torch.sigmoid(covid_out).cpu().numpy())
            all_labels.extend(labels.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)

    # Overall metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels.astype(int), preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print('\n' + '=' * 60)
    print('VALIDATION RESULTS')
    print('=' * 60)
    print(f'  Accuracy:    {acc:.4f}')
    print(f'  F1 Score:    {f1:.4f}')
    print(f'  AUC-ROC:     {auc:.4f}')
    print(f'  Sensitivity: {sensitivity:.4f}')
    print(f'  Specificity: {specificity:.4f}')
    print(f'\nConfusion Matrix')
    print(f'  TN={tn}  FP={fp}')
    print(f'  FN={fn}  TP={tp}')
    print(f'\n{classification_report(labels.astype(int), preds, target_names=["Non-COVID", "COVID"])}')

    # Per-source metrics
    per_source, final_score = compute_per_source_metrics(val_data, probs, labels)
    print('PER-SOURCE METRIC')
    for sid, r in sorted(per_source.items()):
        print(f'  Source {sid}: {r["n_scans"]} scans | '
              f'F1_COVID={r["f1_covid"]:.4f} | '
              f'F1_NonCOVID={r["f1_noncovid"]:.4f} | '
              f'Avg={r["avg"]:.4f}')
    print(f'\nCOMPETITION SCORE = {final_score:.4f}')

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'F1: {f1:.4f}\n')
            f.write(f'AUC: {auc:.4f}\n')
            f.write(f'Sensitivity: {sensitivity:.4f}\n')
            f.write(f'Specificity: {specificity:.4f}\n')
            f.write(f'Competition Score: {final_score:.4f}\n')
            for sid, r in sorted(per_source.items()):
                f.write(f'Source {sid}: F1_COVID={r["f1_covid"]:.4f} '
                        f'F1_NonCOVID={r["f1_noncovid"]:.4f} Avg={r["avg"]:.4f}\n')
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate COVID-19 CT checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results text file')
    args = parser.parse_args()
    main(args)
