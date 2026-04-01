"""
Train multi-task EfficientNet-B7 for multi-source COVID-19 detection.

Usage:
    python train.py --gamma 0.5 --epochs 8 --data_dir /path/to/preprocessed
"""

import os
import copy
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import pandas as pd
from collections import Counter

from src import (
    MultiTaskEfficientNet,
    build_criteria,
    CovidCTDataset,
    get_train_transforms,
    get_val_transforms,
    train_one_epoch,
    validate,
    compute_per_source_metrics,
)


# ── Per-source scan counts from the training set ──────────────────────
# Source 0: 328, Source 1: 330, Source 2: 330, Source 3: 234
SOURCE_COUNTS = [328, 330, 330, 234]
SOURCE_TOTAL = sum(SOURCE_COUNTS)
SOURCE_FREQS = [c / SOURCE_TOTAL for c in SOURCE_COUNTS]


def build_scanlist(base_path, source_map, prefix=''):
    """Build a list of scan dicts from the preprocessed directory."""
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


def get_source_map(covid_csv, noncovid_csv):
    """Build scan_name -> data_centre mapping from CSV files."""
    source_map = {}
    for _, row in covid_csv.iterrows():
        source_map[row['ct_scan_name']] = int(row['data_centre'])
    for _, row in noncovid_csv.iterrows():
        source_map[row['ct_scan_name']] = int(row['data_centre'])
    return source_map


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load source maps ──────────────────────────────────────────────
    train_covid = pd.read_csv(os.path.join(args.csv_dir, 'train_covid.csv'))
    train_noncovid = pd.read_csv(os.path.join(args.csv_dir, 'train_non_covid.csv'))
    val_covid = pd.read_csv(os.path.join(args.csv_dir, 'validation_covid.csv'))
    val_noncovid = pd.read_csv(os.path.join(args.csv_dir, 'validation_non_covid.csv'))

    train_srcmap = get_source_map(train_covid, train_noncovid)
    val_srcmap = get_source_map(val_covid, val_noncovid)

    # ── Build scan lists ──────────────────────────────────────────────
    train_data = build_scanlist(
        os.path.join(args.data_dir, 'train'), train_srcmap
    )
    val_data = build_scanlist(
        os.path.join(args.data_dir, 'val'), val_srcmap, prefix='val_'
    )
    print(f'Train: {len(train_data)} | Val: {len(val_data)}')
    print(f'Train sources: {dict(sorted(Counter(s["source"] for s in train_data).items()))}')

    # ── Data loaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        CovidCTDataset(train_data, get_train_transforms()),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        CovidCTDataset(val_data, get_val_transforms()),
        batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model + optimiser ─────────────────────────────────────────────
    model = MultiTaskEfficientNet(pretrained=True, num_sources=4).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    covid_criterion, source_criterion = build_criteria(SOURCE_FREQS, device)
    scaler = GradScaler()

    print(f'Source frequencies: {[f"{f:.4f}" for f in SOURCE_FREQS]}')
    print(f'LA log-priors: {source_criterion.log_priors}')
    print(f'Training with gamma={args.gamma} for {args.epochs} epochs')
    print('=' * 60)

    # ── Training loop ─────────────────────────────────────────────────
    best_f1, best_state = 0.0, None

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer,
            covid_criterion, source_criterion, scaler, device, args.gamma,
        )
        val_loss, val_f1, val_auc, val_probs, val_labels = validate(
            model, val_loader,
            covid_criterion, source_criterion, device, args.gamma,
        )

        print(f'  Train  loss={train_loss:.4f}  F1={train_f1:.4f}')
        print(f'  Val    loss={val_loss:.4f}  F1={val_f1:.4f}  AUC={val_auc:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            print('  >> New best model')

    # ── Final evaluation ──────────────────────────────────────────────
    model.load_state_dict(best_state)
    _, val_f1, val_auc, val_probs, val_labels = validate(
        model, val_loader,
        covid_criterion, source_criterion, device, args.gamma,
    )
    per_source, final_score = compute_per_source_metrics(val_data, val_probs, val_labels)

    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)
    print(f'F1={val_f1:.4f}  AUC={val_auc:.4f}  Final Score={final_score:.4f}')
    for sid, r in sorted(per_source.items()):
        print(f'  Source {sid}: {r["n_scans"]} scans | '
              f'F1_COVID={r["f1_covid"]:.4f} | '
              f'F1_NonCOVID={r["f1_noncovid"]:.4f} | '
              f'Avg={r["avg"]:.4f}')

    # ── Save checkpoint ───────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'best_gamma{args.gamma}.pth')
    torch.save(best_state, save_path)
    print(f'\nCheckpoint saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-task COVID-19 CT Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed data (contains train/ and val/)')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Path to directory with train/val CSV files')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    main(args)
