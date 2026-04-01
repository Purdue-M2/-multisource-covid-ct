# Robust Multi-Source COVID-19 Detection in CT Images


<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
  <a href="#pretrained-weights"><img src="https://img.shields.io/badge/Weights-Checkpoint-blue" alt="Weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1-orange" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="#">[Paper]</a> •
  <a href="#getting-started">[Getting Started]</a> •
  <a href="#pretrained-weights">[Weights]</a> •
  <a href="#quick-start-colab">[Colab]</a> •
  <a href="#citation">[Citation]</a>
</p>

> **Asmita Yuki Pritha**\*, **Jason Xu**\*, **Daniel Ding**\*, **Justin Li**\*, Aryana Hou, Xin Wang, Shu Hu†
>
> *Equal contribution, †Corresponding author
>
> M2 Lab, Purdue University

---

## Overview

We propose a multi-task learning framework that pairs binary COVID-19 diagnosis with auxiliary source (hospital) identification over a shared **EfficientNet-B7** backbone. A **logit-adjusted cross-entropy loss** on the source head corrects for uneven hospital contributions, pushing the shared encoder toward source-invariant representations.

<p align="center">
  <img src="fig/figurepipeline1.png" width="90%"/>
</p>

## Key Results

| Configuration | γ | F1 | AUC | Accuracy | Competition Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline (BCE only) | — | 0.8915 | 0.9627 | — | 0.8008 |
| Multi-task + CE | 1.0 | 0.8930 | 0.9715 | 0.9058 | 0.7942 |
| Multi-task + LA | 0.1 | 0.8861 | 0.9656 | 0.9123 | 0.7988 |
| Multi-task + LA | 0.2 | 0.8794 | 0.9561 | 0.8994 | 0.7850 |
| **Multi-task + LA (ours)** | **0.5** | **0.9098** | **0.9647** | **0.9253** | **0.8194** |
| Multi-task + LA | 1.0 | 0.8800 | 0.9462 | 0.8929 | 0.7910 |

<details>
<summary><b>Per-source breakdown (γ = 0.5)</b></summary>

| Source | Scans | F1 COVID | F1 Non-COVID | Avg |
|:---|:---:|:---:|:---:|:---:|
| Source 0 | 90 | 0.9032 | 0.8966 | 0.8999 |
| Source 1 | 90 | 0.8571 | 0.8750 | 0.8661 |
| Source 2 | 83 | 0.9459 | 0.9565 | 0.9512 |
| Source 3 | 45 | 0.0000 | 0.8889 | 0.4444 |

*Note: Source 3 validation set contains 0 COVID samples, making F1_COVID undefined.*

</details>

## Getting Started

### 1. Environment

```bash
git clone https://github.com/Purdue-M2/multisource-covid-ct.git
cd multisource-covid-ct
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 12.1 on a single NVIDIA A100.

### 2. Data Preparation

Download the [PHAROS Multi-Source COVID-19 dataset](https://pharos.aimlab.app/) and place files following the structure in [`data/README.md`](data/README.md).

### 3. Preprocessing

```bash
python preprocess.py \
    --raw_dir data/raw \
    --output_dir data/preprocessed
```

This applies SSFL lung extraction and KDS sampling (8 slices/scan, 256×256).

### 4. Training

```bash
python train.py \
    --data_dir data/preprocessed \
    --csv_dir data/ \
    --gamma 0.5 \
    --epochs 8 \
    --batch_size 10
```

To sweep γ:

```bash
bash scripts/sweep_gamma.sh data/preprocessed data/
```

### 5. Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_gamma0.5.pth \
    --data_dir data/preprocessed \
    --csv_dir data/ \
    --output results.txt
```

### 6. Inference on New Data

```bash
python inference.py \
    --checkpoint checkpoints/best_gamma0.5.pth \
    --data_dir data/preprocessed/test \
    --output submission.csv
```

### 7. Reproduce Figures

```bash
python scripts/visualize_results.py --output_dir fig/
```

This generates `gamma_sensitivity.pdf`, `per_source_f1.pdf`, and `gamma_comparison.pdf`.

## Pretrained Weights

| Model | γ | Score | Download |
|:---|:---:|:---:|:---|
| EfficientNet-B7 + LA | 0.5 | 0.8194 | [GitHub Release](https://github.com/Purdue-M2/multisource-covid-ct/releases) |

Download the checkpoint and place it in `checkpoints/`:

```bash
mkdir -p checkpoints
# Download from GitHub Releases and place here
mv best_gamma0.5.pth checkpoints/
```

### Quick Start (Colab)

For a one-click end-to-end run, open **`MultiSource_COVID_CT.ipynb`** in Google Colab with an A100 GPU runtime. The notebook handles data extraction, preprocessing, training, evaluation, and submission generation.

## Project Structure

```
├── configs/
│   └── default.yaml              # Hyperparameters and augmentation config
├── data/
│   └── README.md                 # Dataset download and setup instructions
├── fig/
│   └── figurepipeline1.png       # Pipeline overview figure
├── scripts/
│   ├── sweep_gamma.sh            # γ sweep script
│   └── visualize_results.py      # Generate paper figures
├── src/
│   ├── __init__.py
│   ├── model.py                  # Multi-task EfficientNet-B7
│   ├── losses.py                 # BCE + Logit-Adjusted CE
│   ├── dataset.py                # CT dataset class and augmentations
│   ├── preprocessing.py          # Lung extraction + KDS sampling
│   └── engine.py                 # Training and evaluation loops
├── MultiSource_COVID_CT.ipynb    # End-to-end Colab notebook
├── train.py                      # Main training script
├── preprocess.py                 # Data preprocessing script
├── evaluate.py                   # Standalone evaluation script
├── inference.py                  # Inference on new data
├── requirements.txt
├── LICENSE
└── README.md
```

## Method

1. **Preprocessing** — SSFL lung extraction isolates the lung region via spatial filtering, binarization, and morphological closing. KDS fits a Gaussian KDE over slice-level lung areas and selects 8 representative slices per scan.

2. **Architecture** — EfficientNet-B7 processes each slice independently, producing 8 feature vectors of dimension 2560. Element-wise mean pooling aggregates them into a single scan-level representation, which feeds two heads: a binary COVID-19 classifier and a 4-class source identifier.

3. **Loss** — The COVID head uses BCE. The source head uses logit-adjusted cross-entropy, which adds log-frequency offsets before softmax to correct for uneven hospital contributions. The combined loss is ℓ = ℓ\_CE + γ · ℓ\_LA.

## Citation

```bibtex
@inproceedings{pritha2026multisource,
  title={Robust Multi-Source COVID-19 Detection in CT Images},
  author={Pritha, Asmita Yuki and Xu, Jason and Ding, Daniel and Li, Justin and Hou, Aryana and Wang, Xin and Hu, Shu},
  booktitle={CVPR 2026 Workshop on PHAROS},
  year={2026}
}
```

## Acknowledgements

This work is supported by the U.S. National Science Foundation (NSF) under grant IIS-2434967, and the National Artificial Intelligence Research Resource (NAIRR) Pilot and TACC Lonestar6.

## License

This project is released under the [MIT License](LICENSE).
