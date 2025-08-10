#!/usr/bin/env python3
"""
compute_size_thresholds.py

Compute per-size masking thresholds from existing size-by-SOA curves in
data/masking_curves_all_sizes.csv using a smoothing spline and a 0.5 accuracy
criterion. Outputs a CSV and a summary plot under runs/size_thresholds/.

Threshold definition: SOA at which the smoothed accuracy crosses 0.5.
If no exact crossing exists within the observed SOA range, pick the SOA
within-range where |accuracy-0.5| is minimized.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def compute_threshold_from_curve(soa: np.ndarray, acc: np.ndarray, smoothing: float = 0.01, grid: int = 2001):
    # Sort by SOA to ensure monotonic x
    order = np.argsort(soa)
    x = np.asarray(soa, dtype=float)[order]
    y = np.asarray(acc, dtype=float)[order]

    # Guard against duplicate x by adding tiny jitter if needed
    if len(np.unique(x)) < len(x):
        eps = 1e-6
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                x[i] = x[i] + eps

    # Fit a smoothing spline
    try:
        spl = UnivariateSpline(x, y, s=smoothing)
    except Exception:
        # Fall back to linear interp if spline fails
        def lin_interp(xx):
            return np.interp(xx, x, y)
        spl = lin_interp

    # Dense grid within observed range
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xx = np.linspace(xmin, xmax, grid)
    yy = spl(xx) if callable(spl) else spl(xx)

    # Find crossing near 0.5
    target = 0.5
    diff = yy - target
    sign = np.sign(diff)
    sign[sign == 0] = 1
    cross_idx = np.where(np.diff(sign) != 0)[0]

    if len(cross_idx) > 0:
        # Choose the first crossing; refines by linear interpolation
        i = int(cross_idx[0])
        x1, x2 = xx[i], xx[i+1]
        y1, y2 = yy[i], yy[i+1]
        if y2 != y1:
            thr = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        else:
            thr = (x1 + x2) / 2.0
        method = 'crossing'
    else:
        # Pick closest point to 0.5 within range
        j = int(np.argmin(np.abs(diff)))
        thr = float(xx[j])
        method = 'closest'

    # Also return minimum accuracy location as a sanity check
    jmin = int(np.argmin(yy))
    return {
        'threshold': float(thr),
        'method': method,
        'min_acc': float(yy[jmin]),
        'min_acc_soa': float(xx[jmin]),
    }


def main():
    in_csv = os.path.join('data', 'masking_curves_all_sizes.csv')
    out_dir = os.path.join('runs', 'size_thresholds')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)
    required_cols = {'N_nodes', 'SOA', 'accuracy'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    rows = []

    plt.figure(figsize=(7.5, 5.0))
    for i, (N, g) in enumerate(sorted(df.groupby('N_nodes'), key=lambda kv: kv[0])):
        soa = g['SOA'].to_numpy()
        acc = g['accuracy'].to_numpy()
        res = compute_threshold_from_curve(soa, acc, smoothing=0.01, grid=2001)
        rows.append({
            'N_nodes': int(N),
            'threshold': res['threshold'],
            'method': res['method'],
            'min_acc': res['min_acc'],
            'min_acc_soa': res['min_acc_soa'],
        })

        # Plot curves and the estimated threshold
        order = np.argsort(soa)
        x = soa[order]
        y = acc[order]
        try:
            spl = UnivariateSpline(x, y, s=0.01)
            xx = np.linspace(np.min(x), np.max(x), 1001)
            yy = spl(xx)
            plt.plot(xx, yy, '-', label=f'N={N}')
        except Exception:
            plt.plot(x, y, '-', label=f'N={N}')
        plt.plot(x, y, 'o', alpha=0.7)
        plt.axvline(res['threshold'], color=plt.gca().lines[-1].get_color(), ls='--', alpha=0.5)

    out_csv = os.path.join(out_dir, 'size_thresholds.csv')
    out_png = os.path.join(out_dir, 'size_thresholds.png')

    pd.DataFrame(rows).sort_values('N_nodes').to_csv(out_csv, index=False)

    plt.axhline(0.5, color='k', ls=':', alpha=0.6)
    plt.xlabel('SOA')
    plt.ylabel('Accuracy')
    plt.title('Size curves and 0.5-threshold estimates')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Quick printout
    print(pd.read_csv(out_csv).to_string(index=False))


if __name__ == '__main__':
    main()
