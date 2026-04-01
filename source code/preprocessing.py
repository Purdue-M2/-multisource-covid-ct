"""
SSFL Lung Extraction + Kernel-Density-based Slice Sampling (KDS).

Preprocesses raw CT scans by:
1. Isolating the lung region in each 2D slice via spatial filtering,
   binarization, and morphological closing.
2. Selecting 8 representative slices per scan using KDS, which fits
   a Gaussian KDE over lung areas and partitions the CDF into equal
   percentile bins.
"""

import os
import numpy as np
import cv2
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.stats import gaussian_kde
from tqdm import tqdm


def is_valid_image(filename):
    if filename.startswith('._') or filename.startswith('.'):
        return False
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))


def spatial_filtering(image, kernel_size=3):
    mean_filtered = uniform_filter(image.astype(np.float64), size=kernel_size)
    min_filtered = minimum_filter(mean_filtered, size=kernel_size)
    return min_filtered.astype(np.uint8)


def extract_lung(image, threshold=None):
    filtered = spatial_filtering(image, kernel_size=3)
    if threshold is None:
        threshold, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(threshold * 0.5, 20)
    mask = np.zeros_like(filtered, dtype=np.uint8)
    mask[filtered >= threshold] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    return filled, filtered


def crop_lung(image, mask, target_size=256):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return cv2.resize(image, (target_size, target_size)), 0
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    pad = 5
    y_min = max(0, y_min - pad)
    y_max = min(image.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(image.shape[1], x_max + pad)
    cropped = image[y_min:y_max, x_min:x_max]
    return cv2.resize(cropped, (target_size, target_size)), np.sum(mask)


def kds_sampling(scan_slices, n_samples=8):
    """Kernel-Density-based Slice Sampling.

    Fits a Gaussian KDE over lung areas across all valid slices,
    partitions the CDF into ``n_samples`` equal percentile bins,
    and selects the slice nearest to each bin midpoint.
    """
    areas, indices = [], []
    for i, path in enumerate(scan_slices):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask, _ = extract_lung(img)
        areas.append(np.sum(mask))
        indices.append(i)

    areas = np.array(areas, dtype=np.float64)
    indices = np.array(indices)

    if len(areas) < n_samples:
        return indices.tolist()

    valid = areas > 0
    if np.sum(valid) < n_samples:
        return np.linspace(0, len(scan_slices) - 1, n_samples, dtype=int).tolist()

    valid_areas = areas[valid]
    valid_indices = indices[valid]

    try:
        kde = gaussian_kde(valid_areas, bw_method='scott')
    except Exception:
        sel = np.linspace(0, len(valid_indices) - 1, n_samples, dtype=int)
        return valid_indices[sel].tolist()

    density = kde(valid_areas)
    sort_order = np.argsort(valid_areas)
    sorted_density = density[sort_order]
    cdf = np.cumsum(sorted_density)
    cdf = cdf / cdf[-1]

    picked = []
    for i in range(n_samples):
        lo, hi = i / n_samples, (i + 1) / n_samples
        mid = (lo + hi) / 2
        interval = np.where((cdf >= lo) & (cdf < hi))[0]
        if len(interval) == 0:
            picked.append(np.argmin(np.abs(cdf - mid)))
        else:
            best = interval[np.argmin(np.abs(cdf[interval] - mid))]
            picked.append(best)

    picked = list(dict.fromkeys(picked))
    while len(picked) < n_samples:
        for idx in np.linspace(0, len(sort_order) - 1, n_samples * 2, dtype=int):
            if idx not in picked:
                picked.append(idx)
                if len(picked) == n_samples:
                    break
    picked = picked[:n_samples]
    final = valid_indices[sort_order[picked]]
    return sorted(final.tolist())


def preprocess_scans(scans, output_dir, split_name):
    """Run lung extraction + KDS on a list of scans and save to disk."""
    os.makedirs(os.path.join(output_dir, 'covid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'non-covid'), exist_ok=True)

    processed = []
    for scan in tqdm(scans, desc=split_name):
        selected = kds_sampling(scan['slices'], n_samples=8)
        images = []
        for idx in selected:
            img = cv2.imread(scan['slices'][idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            mask, _ = extract_lung(img)
            cropped, _ = crop_lung(img, mask)
            images.append(cropped)

        if len(images) == 0:
            continue
        while len(images) < 8:
            images.append(images[-1])

        save_dir = os.path.join(output_dir, scan['label_name'], scan['scan_id'])
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(images[:8]):
            cv2.imwrite(os.path.join(save_dir, f'slice_{i:02d}.png'), img)

        processed.append({
            'scan_id': scan['scan_id'],
            'label': scan['label'],
            'label_name': scan['label_name'],
            'path': save_dir,
        })

    print(f'{split_name}: {len(processed)} scans preprocessed')
    return processed
