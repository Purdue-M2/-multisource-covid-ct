#!/bin/bash
# Sweep gamma values for source loss weight
# Usage: bash scripts/sweep_gamma.sh /path/to/preprocessed /path/to/csvs

DATA_DIR=${1:?"Usage: $0 <data_dir> <csv_dir>"}
CSV_DIR=${2:?"Usage: $0 <data_dir> <csv_dir>"}

for gamma in 0.1 0.2 0.5 1.0; do
    echo "============================================"
    echo "Training with gamma=${gamma}"
    echo "============================================"
    python train.py \
        --data_dir "$DATA_DIR" \
        --csv_dir "$CSV_DIR" \
        --gamma "$gamma" \
        --epochs 8 \
        --batch_size 10
done
