# Data

This project uses the **Multi-Source COVID-19 Detection Database** from the
[PHAROS Challenge](https://pharos.aimlab.app/), which aggregates 3D chest CT
scans from four institutionally distinct medical centres.

## Obtaining the Data

The dataset is available through the PHAROS competition page. After downloading,
place the files as follows:

```
data/
├── raw/
│   ├── covid1/
│   ├── covid2/
│   ├── non-covid1/
│   ├── non-covid2/
│   ├── non-covid3/
│   └── validation/
│       └── val/
│           ├── covid/
│           └── non-covid/
├── train_covid.csv
├── train_non_covid.csv
├── validation_covid.csv
└── validation_non_covid.csv
```

## Preprocessing

Run the preprocessing script to apply lung extraction and KDS:

```bash
python preprocess.py --raw_dir data/raw --output_dir data/preprocessed
```

This produces 8 KDS-selected slices per scan at 256×256 resolution.
