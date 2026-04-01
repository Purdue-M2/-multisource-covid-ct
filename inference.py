"""
Run inference on new CT scans and generate a submission CSV.

Usage:
    python inference.py \
        --checkpoint checkpoints/best_gamma0.5.pth \
        --data_dir data/preprocessed/test \
        --output submission.csv
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src import MultiTaskEfficientNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InferenceDataset(Dataset):
    """Dataset for inference on preprocessed scans (no labels needed)."""

    def __init__(self, scan_dirs):
        self.scan_dirs = scan_dirs
        self.transform = Compose([
            Resize(256, 256),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.scan_dirs)

    def __getitem__(self, idx):
        scan_path, scan_id = self.scan_dirs[idx]
        files = sorted([
            f for f in os.listdir(scan_path)
            if f.endswith('.png') and not f.startswith('.')
        ])

        images = []
        for f in files[:8]:
            img = cv2.imread(os.path.join(scan_path, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = self.transform(image=img)['image']
            images.append(img)

        while len(images) < 8:
            images.append(images[-1])

        slices = torch.stack(images[:8], dim=0)
        return slices, scan_id


def collect_scans(data_dir):
    """Collect scan directories for inference."""
    scan_dirs = []
    for root, dirs, files in os.walk(data_dir):
        pngs = [f for f in files if f.endswith('.png')]
        if pngs:
            scan_id = os.path.basename(root)
            scan_dirs.append((root, scan_id))
    return sorted(scan_dirs, key=lambda x: x[1])


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Collect scans
    scan_dirs = collect_scans(args.data_dir)
    print(f'Found {len(scan_dirs)} scans')

    if len(scan_dirs) == 0:
        print('No scans found. Check --data_dir path.')
        return

    # Load model
    model = MultiTaskEfficientNet(pretrained=False, num_sources=4).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Inference
    dataset = InferenceDataset(scan_dirs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    results = []
    with torch.no_grad():
        for images, scan_ids in loader:
            images = images.to(device)
            with autocast():
                covid_out, _ = model(images)
            prob = torch.sigmoid(covid_out).cpu().item()
            pred = 1 if prob >= 0.5 else 0
            results.append({
                'scan_id': scan_ids[0],
                'probability': prob,
                'prediction': pred,
            })

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'\nPredictions saved to {args.output}')
    print(f'  COVID: {sum(r["prediction"] for r in results)}')
    print(f'  Non-COVID: {sum(1 - r["prediction"] for r in results)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on new CT scans')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing preprocessed scan folders')
    parser.add_argument('--output', type=str, default='submission.csv')
    args = parser.parse_args()
    main(args)
