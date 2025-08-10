#!/usr/bin/env python3
"""
compare_structure_vs_size.py

Create a side-by-side comparison of masking thresholds driven by
- Communication structure (small_world rewiring p)
- Network size (N_nodes)

Inputs:
- runs/rewire_sweep_dense/rewire_sweep_thresholds_current.csv
- runs/size_thresholds/size_thresholds.csv

Outputs:
- runs/size_thresholds/structure_vs_size_thresholds.png
- runs/size_thresholds/structure_vs_size_thresholds.csv (tidy summary)
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_structure(thr_csv: str) -> pd.DataFrame:
    df = pd.read_csv(thr_csv)
    # Keep aggregated rows only (have Threshold_mean)
    keep = [c for c in ['RewireP','Size','Seeds','Threshold_mean'] if c in df.columns]
    df = df[keep].dropna(subset=['Threshold_mean']).copy()
    # Drop obvious ceilings/sentinels
    ceilings = {16.0, 48.0, 64.0, 96.0}
    df = df[~df['Threshold_mean'].isin(list(ceilings))]
    # Keep finite
    df = df[np.isfinite(df['Threshold_mean'])]
    # Sort by RewireP
    if 'RewireP' in df.columns:
        df = df.sort_values('RewireP')
    return df


def load_size(size_csv: str) -> pd.DataFrame:
    df = pd.read_csv(size_csv)
    need = ['N_nodes','threshold']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in size csv: {missing}")
    # Keep finite
    df = df[np.isfinite(df['threshold'])]
    return df.sort_values('N_nodes')


def main():
    # Ensure a clean plotting state when called repeatedly (e.g., from REPL)
    plt.close('all')
    plt.style.use('default')
    struct_csv = os.path.join('runs','rewire_sweep_dense','rewire_sweep_thresholds_current.csv')
    size_csv = os.path.join('runs','size_thresholds','size_thresholds.csv')
    out_dir = os.path.join('runs','size_thresholds')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(struct_csv):
        raise FileNotFoundError(struct_csv)
    if not os.path.exists(size_csv):
        raise FileNotFoundError(size_csv)

    df_s = load_structure(struct_csv)
    df_n = load_size(size_csv)

    # Extract best (minimum) structure threshold for headline comparison
    best_row = df_s.loc[df_s['Threshold_mean'].idxmin()] if not df_s.empty else None
    summary_rows = []
    if best_row is not None:
        summary_rows.append({'type':'structure_best', 'x':'p={:.3f}'.format(best_row['RewireP']), 'value':float(best_row['Threshold_mean'])})

    for _, r in df_n.iterrows():
        summary_rows.append({'type':'size', 'x':'N={}'.format(int(r['N_nodes'])), 'value':float(r['threshold'])})

    tidy = pd.DataFrame(summary_rows)
    tidy_csv = os.path.join(out_dir, 'structure_vs_size_thresholds.csv')
    tidy.to_csv(tidy_csv, index=False)

    # Plot: two panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Left: thresholds vs rewire p (filtered)
    ax = axes[0]
    if not df_s.empty:
        ax.plot(df_s['RewireP'], df_s['Threshold_mean'], 'o-', label='Threshold (mean)')
        ax.set_xlabel('Rewire probability p')
        ax.set_ylabel('Threshold (SOA)')
        if best_row is not None:
            ax.axvline(float(best_row['RewireP']), color='tab:green', ls='--', alpha=0.6, label='min p')
            ax.axhline(float(best_row['Threshold_mean']), color='tab:green', ls=':', alpha=0.6)
        ax.set_title('Structure (filtered, non-ceiling)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No non-ceiling structure points', ha='center', va='center')
        ax.set_axis_off()

    # Right: threshold vs N
    ax = axes[1]
    ax.plot(df_n['N_nodes'], df_n['threshold'], 'o-', color='tab:orange')
    ax.set_xlabel('Network size (N)')
    ax.set_ylabel('Threshold (SOA)')
    ax.set_title('Size thresholds (0.5 crossing)')

    out_png = os.path.join(out_dir, 'structure_vs_size_thresholds.png')
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    # Print concise headline to stdout
    if best_row is not None:
        print(f"Best structure threshold: p={best_row['RewireP']:.3f}, thr={best_row['Threshold_mean']:.3f}")
    print('Saved:', out_png)
    print('Saved:', tidy_csv)


if __name__ == '__main__':
    main()
