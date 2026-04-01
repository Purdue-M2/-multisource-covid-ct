"""
Generate publication-quality figures from experimental results.

Produces:
    - fig/gamma_sensitivity.pdf    : Competition score vs gamma
    - fig/per_source_f1.pdf        : Per-source F1 bar chart for best gamma
    - fig/gamma_comparison.pdf     : Per-source breakdown across all gammas

Usage:
    python scripts/visualize_results.py --output_dir fig/
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Experimental results ─────────────────────────────────────────────────

# Baseline: BCE only (no multi-task)
BASELINE = {
    'label': 'Baseline (BCE)',
    'f1': 0.8915, 'auc': 0.9627, 'score': 0.8008,
}

# Multi-task + CE (no logit adjustment), gamma=1.0
MULTITASK_CE = {
    'label': 'MT + CE (γ=1.0)',
    'f1': 0.8930, 'auc': 0.9715, 'score': 0.7942,
    'per_source': [0.9667, 0.8541, 0.8674, 0.4886],
}

# Multi-task + LA (logit-adjusted) for each gamma
RESULTS_LA = {
    0.1: {
        'f1': 0.8861, 'auc': 0.9656, 'acc': 0.9123, 'score': 0.7988,
        'sensitivity': 0.8203, 'specificity': 0.9778,
        'per_source_covid':    [0.8642, 0.9070, 0.8986, 0.0000],
        'per_source_noncovid': [0.8889, 0.9149, 0.9278, 0.9888],
        'per_source_avg':      [0.8765, 0.9109, 0.9132, 0.4944],
    },
    0.2: {
        'f1': 0.8794, 'auc': 0.9561, 'acc': 0.8994, 'score': 0.7850,
        'sensitivity': 0.8828, 'specificity': 0.9111,
        'per_source_covid':    [0.9556, 0.9149, 0.7500, 0.0000],
        'per_source_noncovid': [0.9556, 0.9070, 0.8085, 0.9888],
        'per_source_avg':      [0.9556, 0.9109, 0.7793, 0.4944],
    },
    0.5: {
        'f1': 0.9098, 'auc': 0.9647, 'acc': 0.9253, 'score': 0.8194,
        'sensitivity': 0.8828, 'specificity': 0.8944,
        'per_source_covid':    [0.9032, 0.8571, 0.9459, 0.0000],
        'per_source_noncovid': [0.8966, 0.8750, 0.9565, 0.8889],
        'per_source_avg':      [0.8999, 0.8661, 0.9512, 0.4444],
    },
    1.0: {
        'f1': 0.8800, 'auc': 0.9462, 'acc': 0.8929, 'score': 0.7910,
        'sensitivity': 0.9453, 'specificity': 0.8556,
        'per_source_covid':    [0.8866, 0.8817, 0.9487, 0.0000],
        'per_source_noncovid': [0.8675, 0.8736, 0.9545, 0.9157],
        'per_source_avg':      [0.8770, 0.8776, 0.9516, 0.4578],
    },
}


def plot_gamma_sensitivity(output_dir):
    """Competition score and F1 vs gamma."""
    gammas = sorted(RESULTS_LA.keys())
    scores = [RESULTS_LA[g]['score'] for g in gammas]
    f1s = [RESULTS_LA[g]['f1'] for g in gammas]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color1 = '#2196F3'
    color2 = '#FF5722'

    ax1.plot(gammas, scores, 'o-', color=color1, linewidth=2, markersize=8, label='Competition Score')
    ax1.axhline(y=BASELINE['score'], color=color1, linestyle='--', alpha=0.5, label=f'Baseline Score ({BASELINE["score"]:.4f})')
    ax1.set_xlabel('γ (Source Loss Weight)', fontsize=12)
    ax1.set_ylabel('Competition Score', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(gammas)

    ax2 = ax1.twinx()
    ax2.plot(gammas, f1s, 's--', color=color2, linewidth=2, markersize=8, label='F1 Score')
    ax2.axhline(y=BASELINE['f1'], color=color2, linestyle='--', alpha=0.5, label=f'Baseline F1 ({BASELINE["f1"]:.4f})')
    ax2.set_ylabel('F1 Score', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    ax1.set_title('Sensitivity Analysis: γ Sweep', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'gamma_sensitivity.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def plot_per_source_f1(output_dir):
    """Per-source F1 bar chart for best gamma (0.5)."""
    best = RESULTS_LA[0.5]
    sources = ['Source 0', 'Source 1', 'Source 2', 'Source 3']
    x = np.arange(len(sources))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, best['per_source_covid'], width,
                   label='F1 COVID', color='#EF5350', edgecolor='white')
    bars2 = ax.bar(x + width/2, best['per_source_noncovid'], width,
                   label='F1 Non-COVID', color='#42A5F5', edgecolor='white')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Source F1 Scores (γ = 0.5, LA Loss)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=best['score'], color='gray', linestyle=':', alpha=0.6,
               label=f'Comp. Score = {best["score"]:.4f}')

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.annotate('Source 3: 0 COVID\nsamples in val set',
                xy=(3, 0.02), fontsize=8, color='red', ha='center',
                style='italic')

    fig.tight_layout()
    path = os.path.join(output_dir, 'per_source_f1.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def plot_gamma_comparison(output_dir):
    """Per-source average F1 across all gammas."""
    gammas = sorted(RESULTS_LA.keys())
    sources = ['Source 0', 'Source 1', 'Source 2', 'Source 3']
    x = np.arange(len(sources))
    n = len(gammas)
    width = 0.18
    colors = ['#66BB6A', '#42A5F5', '#FFA726', '#EF5350']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, g in enumerate(gammas):
        vals = RESULTS_LA[g]['per_source_avg']
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f'γ={g}', color=colors[i],
                      edgecolor='white')

    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title('Per-Source Performance Across γ Values', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    path = os.path.join(output_dir, 'gamma_comparison.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    plot_gamma_sensitivity(args.output_dir)
    plot_per_source_f1(args.output_dir)
    plot_gamma_comparison(args.output_dir)
    print(f'\nAll figures saved to {args.output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='fig/')
    args = parser.parse_args()
    main(args)
