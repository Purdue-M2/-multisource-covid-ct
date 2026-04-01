"""
Run SSFL lung extraction + KDS preprocessing on raw CT scans.

Usage:
    python preprocess.py --raw_dir /path/to/extracted --output_dir /path/to/preprocessed
"""

import os
import argparse
from src.preprocessing import is_valid_image, preprocess_scans


def collect_scans(sources, base_path, split_name):
    scans = []
    for label_name, folders in sources.items():
        label = 1 if label_name == 'covid' else 0
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            if not os.path.exists(folder_path):
                continue
            for scan_name in sorted(os.listdir(folder_path)):
                scan_path = os.path.join(folder_path, scan_name)
                if not os.path.isdir(scan_path) or scan_name.startswith('_'):
                    continue
                slices = sorted([
                    os.path.join(scan_path, f)
                    for f in os.listdir(scan_path) if is_valid_image(f)
                ])
                if len(slices) >= 5:
                    scans.append({
                        'scan_id': f'{folder}_{scan_name}',
                        'slices': slices,
                        'label': label,
                        'label_name': label_name,
                    })
    print(f'{split_name}: {len(scans)} scans '
          f'(COVID={sum(1 for s in scans if s["label"] == 1)}, '
          f'Non-COVID={sum(1 for s in scans if s["label"] == 0)})')
    return scans


def collect_val_scans(val_base):
    scans = []
    for label_name in ['covid', 'non-covid']:
        label = 1 if label_name == 'covid' else 0
        label_path = os.path.join(val_base, label_name)
        if not os.path.exists(label_path):
            continue
        for scan_name in sorted(os.listdir(label_path)):
            scan_path = os.path.join(label_path, scan_name)
            if not os.path.isdir(scan_path) or scan_name.startswith('_'):
                continue
            slices = sorted([
                os.path.join(scan_path, f)
                for f in os.listdir(scan_path) if is_valid_image(f)
            ])
            if len(slices) >= 5:
                scans.append({
                    'scan_id': f'val_{scan_name}',
                    'slices': slices,
                    'label': label,
                    'label_name': label_name,
                })
    print(f'Validation: {len(scans)} scans')
    return scans


def main(args):
    train_sources = {
        'covid': ['covid1', 'covid2'],
        'non-covid': ['non-covid1', 'non-covid2', 'non-covid3'],
    }

    train_scans = collect_scans(train_sources, args.raw_dir, 'Train')
    val_scans = collect_val_scans(os.path.join(args.raw_dir, 'validation', 'val'))

    preprocess_scans(train_scans, os.path.join(args.output_dir, 'train'), 'train')
    preprocess_scans(val_scans, os.path.join(args.output_dir, 'val'), 'val')
    print('Preprocessing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw CT scans')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to extracted raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save preprocessed scans')
    args = parser.parse_args()
    main(args)
