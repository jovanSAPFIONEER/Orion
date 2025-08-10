import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

"""
Fit a smooth curve to Threshold_mean vs RewireP and estimate the minimum.
Reads runs/rewire_sweep/rewire_sweep_thresholds.csv (aggregated rows) and writes an overlaid figure.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='runs/rewire_sweep/rewire_sweep_thresholds.csv')
    ap.add_argument('--outdir', type=str, default='runs/rewire_sweep')
    ap.add_argument('--smoothing', type=float, default=0.0, help='Smoothing factor s for UnivariateSpline (0 = interpolate)')
    ap.add_argument('--grid', type=int, default=1001)
    ap.add_argument('--drop_at_or_above', type=float, default=None, help='Drop aggregated rows with Threshold_mean >= this value (ceiling).')
    ap.add_argument('--drop_exact', type=str, default=None, help='Comma-separated exact Threshold_mean values to drop (e.g., "16,24").')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Keep only aggregated rows where Threshold_mean exists
    agg = df.dropna(subset=['Threshold_mean']).copy()
    # Some rows are per-seed with Threshold50_*; aggregated rows have 'Seeds' set
    if 'Seeds' in agg.columns:
        agg = agg[agg['Seeds'].notna()]
    # Optionally drop ceiling-like aggregated points
    if args.drop_exact:
        try:
            exact_vals = {float(x.strip()) for x in args.drop_exact.split(',') if x.strip()}
            agg = agg[~agg['Threshold_mean'].isin(exact_vals)]
        except Exception:
            pass
    if args.drop_at_or_above is not None:
        agg = agg[agg['Threshold_mean'] < float(args.drop_at_or_above)]

    agg = agg.sort_values('RewireP')
    if len(agg) < 4:
        raise SystemExit('Not enough aggregated rewire points to fit a smooth curve (need >=4).')

    x = agg['RewireP'].to_numpy(dtype=float)
    y = agg['Threshold_mean'].to_numpy(dtype=float)

    # Fit smoothing spline
    s = args.smoothing
    try:
        spl = UnivariateSpline(x, y, s=s, k=3)
    except Exception:
        # Fallback: use slight smoothing if duplicates or ill-conditioned
        spl = UnivariateSpline(x, y, s=1e-6 if s == 0 else s, k=3)

    xx = np.linspace(float(x.min()), float(x.max()), args.grid)
    yy = spl(xx)

    # Find minimum on the grid
    i_min = int(np.argmin(yy))
    x_min = float(xx[i_min])
    y_min = float(yy[i_min])

    # Plot overlay
    plt.figure(figsize=(7.5,4.8))
    plt.plot(x, y, 'o-', color='#2a6fdb', label='Mean threshold')
    plt.plot(xx, yy, '-', color='#d34e24', alpha=0.85, label='Spline fit')
    plt.axvline(x_min, color='#d34e24', ls='--', alpha=0.5)
    plt.scatter([x_min], [y_min], color='#d34e24')
    plt.xlabel('Rewire probability (small_world)')
    plt.ylabel('Masking threshold (SOA at 50%)')
    plt.title('Threshold vs Rewire Probability (Spline)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.outdir, 'rewire_threshold_curve_spline.png')
    out_pdf = os.path.join(args.outdir, 'rewire_threshold_curve_spline.pdf')
    plt.savefig(out_png, dpi=200); plt.savefig(out_pdf)

    print(f'Estimated minimum: rewire_p ~ {x_min:.3f}, threshold ~ {y_min:.3f}')
    print('Wrote figures:', out_png, out_pdf)

if __name__ == '__main__':
    main()
