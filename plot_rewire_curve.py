import os, argparse
from typing import cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Plot threshold vs rewire_p and relationships with structural metrics.
Reads runs/rewire_sweep/rewire_sweep_with_metrics.csv and writes PNG/PDF.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='runs/rewire_sweep/rewire_sweep_with_metrics.csv')
    ap.add_argument('--outdir', type=str, default='runs/rewire_sweep')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df: pd.DataFrame = pd.read_csv(args.csv)  # type: ignore[call-overload]

    # Use aggregated rows where Threshold_mean is present
    agg = cast(pd.DataFrame, df.dropna(subset=['Threshold_mean']).copy())
    agg = cast(pd.DataFrame, agg.sort_values('RewireP'))

    # Primary curve: Threshold_mean vs RewireP with CI if present
    plt.figure(figsize=(7.5,4.8))
    x = np.asarray(agg['RewireP'].to_numpy(dtype=float), dtype=float)
    y = np.asarray(agg['Threshold_mean'].to_numpy(dtype=float), dtype=float)
    ylo = np.asarray(agg['Threshold_lo'].to_numpy(dtype=float), dtype=float) if 'Threshold_lo' in agg.columns else None
    yhi = np.asarray(agg['Threshold_hi'].to_numpy(dtype=float), dtype=float) if 'Threshold_hi' in agg.columns else None
    plt.plot(x, y, marker='o', color='#2a6fdb', label='Mean threshold')  # type: ignore[arg-type]
    if ylo is not None and yhi is not None and np.isfinite(cast(np.ndarray, ylo)).any() and np.isfinite(cast(np.ndarray, yhi)).any():
        plt.fill_between(x, cast(np.ndarray, ylo), cast(np.ndarray, yhi), color='#2a6fdb', alpha=0.2, label='95% CI')  # type: ignore[arg-type]
    plt.xlabel('Rewire probability (small_world)')
    plt.ylabel('Masking threshold (SOA at 50%)')
    plt.title('Threshold vs Rewire Probability')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(args.outdir, 'rewire_threshold_curve.png')
    out1_pdf = os.path.join(args.outdir, 'rewire_threshold_curve.pdf')
    plt.savefig(out1, dpi=200); plt.savefig(out1_pdf)
    plt.close()

    # Scatter vs structural metrics (CommMean, Fiedler, LongRangeFrac, WeightedEff)
    metrics = [c for c in ['CommMean','Fiedler','LongRangeFrac','WeightedEff'] if c in agg.columns]
    if metrics:
        n = len(metrics)
        cols = min(3, n)
        rows = int(np.ceil(n/cols))
        plt.figure(figsize=(5*cols, 3.8*rows))
        for i,m in enumerate(metrics, 1):
            ax = plt.subplot(rows, cols, i)
            xm_series = agg[m]
            ym_series = agg['Threshold_mean']
            ax.scatter(xm_series.to_numpy(dtype=float), ym_series.to_numpy(dtype=float), c='#444', s=45)  # type: ignore[arg-type]
            # Fit quadratic for guide
            xm = xm_series.to_numpy(dtype=float); ym = ym_series.to_numpy(dtype=float)
            if len(xm) >= 4 and np.unique(xm).size >= 3:
                try:
                    A = np.vstack([xm**2, xm, np.ones_like(xm)]).T
                    coef, *_ = np.linalg.lstsq(A, ym, rcond=None)
                    xx = np.linspace(xm.min(), xm.max(), 200)
                    yy = coef[0]*xx**2 + coef[1]*xx + coef[2]
                    ax.plot(xx, yy, color='#d34e24', alpha=0.8)  # type: ignore[arg-type]
                except Exception:
                    # Skip guide if ill-conditioned
                    pass
            ax.set_xlabel(m)
            ax.set_ylabel('Threshold')
            ax.set_title(f'Threshold vs {m}')
            ax.grid(alpha=0.25)
        plt.tight_layout()
        out2 = os.path.join(args.outdir, 'threshold_vs_metrics.png')
        out2_pdf = os.path.join(args.outdir, 'threshold_vs_metrics.pdf')
        plt.savefig(out2, dpi=200); plt.savefig(out2_pdf)
        plt.close()

    # Console summary
    if len(agg):
        thr_vals = agg['Threshold_mean'].to_numpy(dtype=float)
        idx_min = int(np.argmin(thr_vals))
        idx_max = int(np.argmax(thr_vals))
        print('Min threshold at rewire_p=', float(agg.iloc[idx_min]['RewireP']), 'thr=', float(agg.iloc[idx_min]['Threshold_mean']))
        print('Max threshold at rewire_p=', float(agg.iloc[idx_max]['RewireP']), 'thr=', float(agg.iloc[idx_max]['Threshold_mean']))
        print('Wrote figures:', out1, out1_pdf, ('and metric plots' if metrics else ''))

if __name__ == '__main__':
    main()
